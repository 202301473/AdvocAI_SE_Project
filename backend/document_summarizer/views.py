import html
import logging
import re
import textwrap
from typing import Any, Dict, List, Tuple

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
from .models import DocumentSession, ChatMessage
from authentication.models import User
import fitz  # PyMuPDF for PDF
from docx import Document
from mongoengine import DoesNotExist
from utils.gemini_client import get_gemini_client, _get_llm_model_name # Import from centralized utility

# Import generalized false positive prevention framework
try:
    from .false_positive_prevention import (
        should_filter_clause,
        validate_category_consistency,
        detect_identical_replacement,
        get_balancing_examples_for_prompt
    )
    FP_FRAMEWORK_AVAILABLE = True
except ImportError:
    FP_FRAMEWORK_AVAILABLE = False

# Import two-stage solution refinement
try:
    from .solution_refinement import (
        refine_clause_solutions_with_patterns_and_llm,
        batch_refine_clauses
    )
    SOLUTION_REFINEMENT_AVAILABLE = True
except ImportError:
    SOLUTION_REFINEMENT_AVAILABLE = False
    logger.warning("Solution refinement module not available")

logger = logging.getLogger(__name__)




def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace for better matching"""
    return re.sub(r'\s+', ' ', text).strip()


def _strip_section_header(text: str) -> str:
    """
    Remove common section headers/numbering from clause text.
    Examples: "10.1", "Section 5", "Article III", etc.
    """
    import re
    # Remove leading section numbers like "10.1", "5.2.3", etc.
    text = re.sub(r'^\s*\d+(\.\d+)*\s*', '', text)
    # Remove "Section X", "Article Y", etc.
    text = re.sub(r'^\s*(Section|Article|Clause|Paragraph)\s+[IVXLCDM\d]+[\.\:)]*\s*', '', text, flags=re.IGNORECASE)
    # Remove standalone headers at start
    text = re.sub(r'^[A-Z][a-z]+\s+\d+(\.\d+)*\s+', '', text)
    return text.strip()


def _find_best_match(full_text: str, snippet: str, min_match_length: int = 30) -> Tuple[int, int]:
    """
    Find the best match for a snippet in the full text using multiple strategies.
    
    Returns: (start_index, end_index) or (-1, -1) if not found
    """
    if not snippet or not full_text:
        return (-1, -1)
    
    snippet = snippet.strip()
    
    # Strategy 1: Exact match
    start_index = full_text.find(snippet)
    if start_index != -1:
        return (start_index, start_index + len(snippet))
    
    # Strategy 2: Case-insensitive exact match
    lower_text = full_text.lower()
    lower_snippet = snippet.lower()
    start_index = lower_text.find(lower_snippet)
    if start_index != -1:
        return (start_index, start_index + len(snippet))
    
    # Strategy 3: Strip section headers and try again
    stripped_snippet = _strip_section_header(snippet)
    if stripped_snippet and len(stripped_snippet) >= min_match_length:
        # Try exact match without section header
        start_index = full_text.find(stripped_snippet)
        if start_index != -1:
            # Find the actual start (may include header in document)
            search_start = max(0, start_index - 50)
            return (search_start, start_index + len(stripped_snippet))
        
        # Try case-insensitive without section header
        start_index = lower_text.find(stripped_snippet.lower())
        if start_index != -1:
            search_start = max(0, start_index - 50)
            return (search_start, start_index + len(stripped_snippet))
    
    # Strategy 4: Normalized whitespace match
    normalized_snippet = _normalize_whitespace(snippet)
    if len(normalized_snippet) < min_match_length:
        return (-1, -1)
    
    # Create normalized version for comparison
    normalized_full = _normalize_whitespace(full_text)
    norm_start = normalized_full.lower().find(normalized_snippet.lower())
    
    if norm_start != -1:
        # Map back to original text position (approximate)
        # Count non-whitespace chars to find position
        char_count = 0
        start_index = 0
        for i, char in enumerate(full_text):
            if not char.isspace():
                char_count += 1
            if char_count >= norm_start:
                start_index = i
                break
        
        # Find end by matching characters
        snippet_chars = [c for c in normalized_snippet.lower() if not c.isspace()]
        matched_chars = 0
        end_index = start_index
        
        for i in range(start_index, len(full_text)):
            if matched_chars >= len(snippet_chars):
                end_index = i
                break
            if not full_text[i].isspace():
                if full_text[i].lower() == snippet_chars[matched_chars]:
                    matched_chars += 1
                else:
                    # Mismatch, try from next position
                    break
        
        if matched_chars >= len(snippet_chars) * 0.9:  # 90% match
            return (start_index, end_index)
    
    # Strategy 5: Try with stripped snippet and normalized whitespace
    if stripped_snippet and len(stripped_snippet) >= min_match_length:
        normalized_stripped = _normalize_whitespace(stripped_snippet)
        norm_start = normalized_full.lower().find(normalized_stripped.lower())
        
        if norm_start != -1:
            char_count = 0
            start_index = 0
            for i, char in enumerate(full_text):
                if not char.isspace():
                    char_count += 1
                if char_count >= norm_start:
                    start_index = max(0, i - 50)  # Include potential header
                    break
            
            snippet_chars = [c for c in normalized_stripped.lower() if not c.isspace()]
            matched_chars = 0
            
            for i in range(start_index, len(full_text)):
                if matched_chars >= len(snippet_chars):
                    end_index = i
                    break
                if not full_text[i].isspace():
                    if full_text[i].lower() == snippet_chars[matched_chars]:
                        matched_chars += 1
            
            if matched_chars >= len(snippet_chars) * 0.85:  # 85% match for stripped
                return (start_index, end_index)
    
    # Strategy 6: Find core phrase (first substantial words, skip section headers)
    snippet_for_words = _strip_section_header(snippet)
    words = [w for w in snippet_for_words.split() if len(w) > 3][:10]  # First 10 significant words
    
    if len(words) >= 3:
        # Try progressively smaller core phrases
        for phrase_length in [7, 5, 4, 3]:
            if len(words) >= phrase_length:
                core_phrase = ' '.join(words[:phrase_length])
                start_index = lower_text.find(core_phrase.lower())
                if start_index != -1:
                    # Extend to reasonable clause boundary
                    end_index = start_index + min(len(snippet), 500)
                    # Try to end at sentence boundary
                    for i in range(end_index, min(end_index + 150, len(full_text))):
                        if full_text[i] in '.!?\n':
                            end_index = i + 1
                            break
                    return (max(0, start_index - 30), end_index)
    
    return (-1, -1)


def _expand_to_sentence_boundary(text: str, start: int, end: int) -> Tuple[int, int]:
    """Expand the range to complete sentence boundaries.
    
    Expands backwards to find the start of the sentence and forwards to find the end.
    This ensures highlighted clauses are complete and coherent.
    """
    # Find sentence start (capital letter after period, or start of text)
    sentence_start = start
    for i in range(start - 1, max(0, start - 200), -1):
        if text[i] in '.!?\n':
            # Found potential sentence end, next sentence starts after whitespace
            sentence_start = i + 1
            while sentence_start < start and text[sentence_start].isspace():
                sentence_start += 1
            break
        # Check for numbered sections like "1.1" or "(a)"
        if i > 0 and text[i].isdigit() and text[i-1] == '\n':
            sentence_start = i
            break
    
    # Find sentence end (period, exclamation, question mark)
    sentence_end = end
    for i in range(end, min(len(text), end + 300)):
        if text[i] in '.!?':
            # Include the punctuation and any trailing space
            sentence_end = i + 1
            while sentence_end < len(text) and text[sentence_end] in ' \t':
                sentence_end += 1
            break
        # Stop at double newline (paragraph break)
        if i < len(text) - 1 and text[i:i+2] == '\n\n':
            sentence_end = i
            break
    
    return (sentence_start, sentence_end)


def _build_highlighted_preview(full_text: str, clauses: List[Dict[str, Any]]) -> Tuple[str, List[int], Dict[int, str]]:
    """Return HTML-safe preview text with risky clauses wrapped in <mark> tags.
    
    Returns:
        Tuple of (highlighted_html, successfully_highlighted_indices, expanded_clause_texts)
        where expanded_clause_texts maps clause_idx -> expanded full sentence text
    """
    if not full_text:
        return "", [], {}

    if not clauses:
        return html.escape(full_text).replace('\n', '<br />'), [], {}

    matches: List[Tuple[int, int, int, int]] = []  # (start, end, risk_score, clause_index)
    successfully_highlighted: List[int] = []  # Track which clause indices were highlighted
    expanded_clause_texts: Dict[int, str] = {}  # Map clause_idx -> expanded sentence text
    lower_text = full_text.lower()

    for clause_idx, clause in enumerate(clauses):
        snippet = (
            clause.get('clause_text')
            or clause.get('clause')
            or clause.get('text')
            or ""
        ).strip()
        if not snippet:
            continue

        risk_score = clause.get('risk_score', 3)
        start_index, end_index = _find_best_match(full_text, snippet)
        
        if start_index == -1:
            # Log with more context for debugging
            stripped = _strip_section_header(snippet)
            logger.warning(
                f"Could not highlight clause (original: '{snippet[:60]}...', "
                f"stripped: '{stripped[:60]}...'). "
                f"Trying partial match as last resort."
            )
            
            # Last resort: find ANY significant portion (at least 50 chars)
            if len(snippet) > 50:
                # Try last 100 chars (often the actual clause content)
                tail = snippet[-100:].strip()
                if len(tail) > 40:
                    start_index = lower_text.find(tail.lower())
                    if start_index != -1:
                        end_index = start_index + len(tail)
                        logger.info(f"Found using tail match: '{tail[:40]}...'")
                    else:
                        # Try middle 100 chars
                        mid_start = len(snippet) // 4
                        middle = snippet[mid_start:mid_start+100].strip()
                        if len(middle) > 40:
                            start_index = lower_text.find(middle.lower())
                            if start_index != -1:
                                end_index = start_index + len(middle)
                                logger.info(f"Found using middle match: '{middle[:40]}...'")
        
        if start_index == -1:
            logger.warning(f"Skipping highlight for clause: {snippet[:80]}...")
            continue
        
        # Expand to complete sentence boundaries for coherent highlighting
        start_index, end_index = _expand_to_sentence_boundary(full_text, start_index, end_index)
        expanded_text = full_text[start_index:end_index].strip()
        expanded_clause_texts[clause_idx] = expanded_text
        logger.info(f"Expanded clause to sentence boundaries: [{start_index}:{end_index}] = '{expanded_text[:60]}...'")

        # Check for overlaps - only merge if they're truly the same clause (>70% overlap)
        # Allow adjacent or slightly overlapping different clauses to coexist
        has_significant_overlap = False
        for i, (existing_start, existing_end, existing_risk, existing_idx) in enumerate(matches):
            overlap_start = max(start_index, existing_start)
            overlap_end = min(end_index, existing_end)
            overlap_length = max(0, overlap_end - overlap_start)
            
            # Calculate overlap percentage relative to both clauses
            current_length = end_index - start_index
            existing_length = existing_end - existing_start
            
            # Check overlap from both perspectives
            overlap_of_current = overlap_length / current_length if current_length > 0 else 0
            overlap_of_existing = overlap_length / existing_length if existing_length > 0 else 0
            
            # Only treat as duplicate if BOTH clauses have >70% overlap
            # This means they're essentially the same clause, not just adjacent
            if overlap_of_current > 0.7 and overlap_of_existing > 0.7:
                # This is a duplicate detection of the same clause - keep higher risk
                if risk_score > existing_risk:
                    # Replace existing with current
                    matches[i] = (start_index, end_index, risk_score, clause_idx)
                    # Update successfully_highlighted and expanded texts
                    if existing_idx in successfully_highlighted:
                        successfully_highlighted.remove(existing_idx)
                    if existing_idx in expanded_clause_texts:
                        del expanded_clause_texts[existing_idx]
                    successfully_highlighted.append(clause_idx)
                    logger.info(f"Duplicate clause detected (overlap {overlap_of_current:.0%}/{overlap_of_existing:.0%}): replacing with higher risk")
                else:
                    # Keep existing, skip current - remove current's expanded text
                    if clause_idx in expanded_clause_texts:
                        del expanded_clause_texts[clause_idx]
                    logger.info(f"Duplicate clause detected (overlap {overlap_of_current:.0%}/{overlap_of_existing:.0%}): keeping existing higher risk")
                has_significant_overlap = True
                break
            elif overlap_length > 0:
                # Some overlap but not duplicates - these are different adjacent clauses
                # Log but allow both to exist
                logger.info(f"Adjacent clauses with minor overlap ({overlap_length} chars): keeping both separate")
        
        if not has_significant_overlap:
            matches.append((start_index, end_index, risk_score, clause_idx))
            successfully_highlighted.append(clause_idx)

    # Sort matches by position
    matches.sort(key=lambda x: x[0])

    if not matches:
        logger.warning(f"No clauses could be highlighted from {len(clauses)} detected risks")
        return html.escape(full_text).replace('\n', '<br />'), []

    highlighted_parts: List[str] = []
    previous_end = 0

    for start_index, end_index, risk_score, clause_idx in matches:
        # Add text before the match
        highlighted_parts.append(html.escape(full_text[previous_end:start_index]))
        
        # Add highlighted match with risk level class
        snippet = full_text[start_index:end_index]
        risk_level = 'high' if risk_score >= 4 else 'medium' if risk_score >= 3 else 'low'
        highlighted_parts.append(
            f"<mark class=\"risk-{risk_level}\" data-risk-score=\"{risk_score}\">{html.escape(snippet)}</mark>"
        )
        previous_end = end_index

    highlighted_parts.append(html.escape(full_text[previous_end:]))
    highlighted_html = ''.join(highlighted_parts)
    
    logger.info(f"Successfully highlighted {len(successfully_highlighted)} out of {len(clauses)} clauses")
    return highlighted_html.replace('\n', '<br />'), list(set(successfully_highlighted)), expanded_clause_texts


def _order_clauses_by_priority(clauses: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
    if not clauses:
        return clauses

    def clause_position(text: str) -> int:
        if not full_text:
            return 0
        idx = full_text.find(text)
        return idx if idx >= 0 else len(full_text)

    return sorted(
        clauses,
        key=lambda item: (
            -int(item.get('risk_score', 3) or 3),
            clause_position(item.get('clause_text', '')),
        )
    )


def _fallback_risk_clauses(full_text: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Enhanced heuristic scan using contextual pattern matching."""
    if not full_text:
        return []

    # Use enhanced risk detector with context awareness
    detected_risks = detect_enhanced_risks(full_text, max_clauses=limit)
    
    # Convert to expected format and add replacement clauses
    enhanced_clauses = []
    for risk in detected_risks:
        category = risk.get('category', 'generic')
        replacement_clause = DEFAULT_REPLACEMENTS.get(category, DEFAULT_REPLACEMENTS['generic'])
        
        enhanced_clauses.append({
            'clause_text': risk['clause_text'],
            'risk_level': risk['risk_level'],
            'risk_score': risk['risk_score'],
            'rationale': risk['rationale'],
            'mitigation': risk['mitigation'],
            'replacement_clause': replacement_clause,
            'confidence': risk.get('confidence', 0.7),
            'category': category,
        })
    
    # Fallback to old method if enhanced detection found nothing
    if not enhanced_clauses:
        logger.warning("Enhanced detection found no risks, using legacy keyword matching")
        lowered = full_text.lower()
        clauses: List[Dict[str, Any]] = []

        for keyword_info in RISK_KEYWORDS[:20]:  # Limit to top 20 patterns
            keyword = keyword_info['pattern']
            matches = list(re.finditer(re.escape(keyword), lowered))
            if matches:
                match = matches[0]  # Just take first match
                start = max(0, match.start() - 220)
                end = min(len(full_text), match.end() + 220)
                snippet = full_text[start:end].strip()
                if snippet:
                    clauses.append({
                        'clause_text': snippet,
                        'risk_level': _score_to_label(keyword_info.get('default_score', 3)),
                        'risk_score': keyword_info.get('default_score', 3),
                        'rationale': keyword_info['rationale'],
                        'mitigation': keyword_info.get('suggestion', ''),
                        'replacement_clause': DEFAULT_REPLACEMENTS.get(
                            keyword_info.get('replacement_category', 'generic'),
                            DEFAULT_REPLACEMENTS['generic']
                        ),
                    })
                if len(clauses) >= limit:
                    break
        return clauses
    
    return enhanced_clauses


def _keyword_score(text: str) -> int:
    if not text:
        return 0
    lowered = text.lower()
    score = 0
    for keyword_info in RISK_KEYWORDS:
        pattern = keyword_info['pattern']
        weight = keyword_info.get('weight', 1)
        score += lowered.count(pattern) * weight
    return score


def _extract_keyword_sentences(full_text: str, max_sentences: int = 12) -> List[str]:
    if not full_text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    scored: List[Tuple[int, int, str]] = []

    for idx, sentence in enumerate(sentences):
        cleaned = sentence.strip()
        if not cleaned:
            continue
        score = _keyword_score(cleaned)
        if score > 0:
            scored.append((score, idx, cleaned))

    if not scored:
        return []

    scored.sort(key=lambda item: (-item[0], item[1]))
    top_sentences = [item[2] for item in scored[:max_sentences]]
    return top_sentences


def _chunk_document(full_text: str, chunk_size: int = 2500, overlap: int = 300) -> List[Dict[str, Any]]:
    """Split document into overlapping chunks to keep model prompts small."""
    if not full_text:
        return []

    if len(full_text) <= chunk_size:
        return [{'text': full_text, 'start': 0, 'end': len(full_text)}]

    chunks: List[Dict[str, Any]] = []
    step = max(chunk_size - overlap, 900)
    position = 0

    while position < len(full_text):
        end = min(len(full_text), position + chunk_size)
        chunk_text = full_text[position:end]
        chunks.append({'text': chunk_text, 'start': position, 'end': end})
        if end == len(full_text):
            break
        position += step

    return chunks


def _dedupe_clauses(clauses: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    """Remove duplicate clause entries while preserving order."""
    seen: Dict[str, int] = {}
    deduped: List[Dict[str, Any]] = []

    for clause in clauses:
        normalized_clause = _normalize_clause_structure(clause)
        clause_text = normalized_clause.get('clause_text')
        if not clause_text:
            continue
        
        # Create fingerprint focusing on significant words
        words = clause_text.lower().split()
        significant_words = [w for w in words if len(w) > 3][:25]
        fingerprint = ' '.join(significant_words)
        
        # Check fingerprint first (fast)
        if fingerprint in seen:
            existing_index = seen[fingerprint]
            existing_clause = deduped[existing_index]
            # Keep higher risk score
            if normalized_clause.get('risk_score', 0) > existing_clause.get('risk_score', 0):
                deduped[existing_index] = normalized_clause
            else:
                # Merge missing attributes
                for key in ('risk_level', 'risk_score', 'rationale', 'mitigation', 'replacement_clause', 'confidence'):
                    if not existing_clause.get(key) and normalized_clause.get(key):
                        existing_clause[key] = normalized_clause.get(key)
            continue
        
        # Check for position overlap if available
        is_duplicate = False
        clause_pos = normalized_clause.get('position')
        
        if clause_pos:
            for i, existing_clause in enumerate(deduped):
                existing_pos = existing_clause.get('position')
                
                # Check document position overlap
                if existing_pos:
                    overlap_start = max(clause_pos[0], existing_pos[0])
                    overlap_end = min(clause_pos[1], existing_pos[1])
                    overlap = max(0, overlap_end - overlap_start)
                    clause_length = clause_pos[1] - clause_pos[0]
                    existing_length = existing_pos[1] - existing_pos[0]
                    
                    # If >50% overlap of either clause, it's same clause
                    if clause_length > 0 and (overlap > clause_length * 0.5 or overlap > existing_length * 0.5):
                        is_duplicate = True
                        # Keep higher risk score
                        if normalized_clause.get('risk_score', 0) > existing_clause.get('risk_score', 0):
                            deduped[i] = normalized_clause
                        break
        
        if is_duplicate:
            continue
        
        seen[fingerprint] = len(deduped)
        deduped.append(normalized_clause)
        
        if len(deduped) >= limit:
            break

    return deduped


def _generate_comprehensive_summary(full_text: str, doc_type: str, llm, doc_type_name: str, use_llm: bool = True) -> Dict[str, Any]:
    """Generate detailed legal document summary with structured sections and plain language explanations.
    
    Args:
        full_text: Full document text
        doc_type: Document type identifier
        llm: LangChain LLM instance
        doc_type_name: Human-readable document type name
        use_llm: Whether to try LLM first (fallback to regex if quota exceeded)
        
    Returns:
        Comprehensive summary dictionary with structured information
    """
    
    # If LLM is disabled or unavailable, use regex extraction directly
    if not use_llm or not llm:
        logger.info("Using regex-based comprehensive summary (LLM disabled)")
        return _generate_comprehensive_summary_from_analysis(
            full_text=full_text,
            doc_type=doc_type,
            doc_type_name=doc_type_name,
            chunk_results=[],
            deduped_clauses=[]
        )
    
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from pydantic import BaseModel, Field
        from typing import Optional
        
        class PartyInfo(BaseModel):
            name: str = Field(..., description="Full legal name of the party")
            role: str = Field(..., description="Role in document (e.g., Employer, Tenant, Service Provider, Disclosing Party)")
            simple_explanation: Optional[str] = Field(None, description="Plain language explanation of what this party does in this agreement")
        
        class FinancialTerm(BaseModel):
            item: str = Field(..., description="What the payment/amount is for")
            amount: str = Field(..., description="The specific amount, rate, or value")
            simple_explanation: Optional[str] = Field(None, description="Plain language explanation")
        
        class TerminationInfo(BaseModel):
            duration: str = Field(..., description="How long the agreement lasts")
            renewal_terms: Optional[str] = Field(None, description="How/when it renews")
            termination_process: str = Field(..., description="How to end the agreement")
            notice_period: Optional[str] = Field(None, description="Required notice to terminate")
            simple_explanation: str = Field(..., description="Plain language explanation of term and exit options")
        
        class LegalTermExplanation(BaseModel):
            term: str = Field(..., description="Complex legal term or phrase")
            meaning: str = Field(..., description="Simple, everyday language explanation")
        
        class ComprehensiveSummary(BaseModel):
            document_type: str = Field(..., description="Specific type of legal document")
            execution_date: Optional[str] = Field(None, description="Date document was/will be signed")
            parties: List[PartyInfo] = Field(..., description="All parties involved with roles")
            purpose: str = Field(..., description="Core reason this document exists (2-3 sentences max)")
            key_obligations: Dict[str, str] = Field(..., description="Map of party name to their main responsibilities")
            financial_terms: List[FinancialTerm] = Field(default_factory=list, description="Payment amounts, fees, compensation")
            term_and_termination: TerminationInfo = Field(..., description="Duration and how to end the agreement")
            compliance_requirements: Optional[List[str]] = Field(default_factory=list, description="Legal compliance obligations")
            important_deadlines: Optional[List[str]] = Field(default_factory=list, description="Time-sensitive obligations")
            attachments_mentioned: Optional[List[str]] = Field(default_factory=list, description="Schedules, exhibits, annexures referenced")
            legal_terms_explained: List[LegalTermExplanation] = Field(default_factory=list, description="Complex legal terms with plain language meanings")
            executive_summary: str = Field(..., description="2-3 paragraph plain language overview suitable for non-lawyers (150-250 words)")
        
        summary_prompt = ChatPromptTemplate.from_messages([
            (
                'system',
                f"You are an expert legal document analyst specializing in {doc_type_name}. "
                "Your role is to create comprehensive, structured summaries that make legal documents "
                "accessible to non-lawyers while maintaining accuracy.\\n\\n"
                "CRITICAL INSTRUCTIONS:\\n"
                "1. Extract ALL key information systematically\\n"
                "2. For complex legal terms, provide plain language explanations\\n"
                "3. Use everyday language in 'simple_explanation' fields\\n"
                "4. Be specific with amounts, dates, timeframes\\n"
                "5. Focus on practical implications for each party\\n"
                "6. The executive_summary should be readable by anyone without legal training\\n\\n"
                "PLAIN LANGUAGE EXAMPLES:\\n"
                "❌ 'Indemnification obligation' → ✅ 'If something goes wrong because of Party A, they must pay for any resulting costs'\\n"
                "❌ 'Force majeure provision' → ✅ 'If unexpected events like natural disasters happen, neither party is blamed'\\n"
                "❌ 'Liquidated damages' → ✅ 'Pre-agreed penalty amount if someone breaks the contract'\\n"
                "❌ 'Representations and warranties' → ✅ 'Promises and guarantees each party is making'\\n\\n"
                "Think of this as explaining the document to a friend who isn't a lawyer."
            ),
            (
                'human',
                "LEGAL DOCUMENT TO ANALYZE:\\n\\n{document_text}\\n\\n"
                "ANALYSIS REQUIREMENTS:\\n\\n"
                "1. DOCUMENT IDENTIFICATION\\n"
                "   - What type of document is this exactly?\\n"
                "   - When was/will it be signed? (check for execution date, effective date)\\n"
                "   - Who are ALL the parties? (get full names and their roles)\\n\\n"
                "2. PURPOSE\\n"
                "   - Why does this document exist?\\n"
                "   - What relationship/transaction does it govern?\\n"
                "   - Write in simple language: 'This agreement allows Party A to... while Party B will...'\\n\\n"
                "3. KEY RIGHTS & OBLIGATIONS\\n"
                "   - For EACH party, what must they do?\\n"
                "   - What are they NOT allowed to do?\\n"
                "   - What do they receive in return?\\n"
                "   - Be specific about deliverables, services, restrictions\\n\\n"
                "4. FINANCIAL TERMS\\n"
                "   - All payment amounts (salary, rent, fees, deposits)\\n"
                "   - When payments are due\\n"
                "   - Penalties, bonuses, incentives\\n"
                "   - Any caps or limits\\n\\n"
                "5. TERM & TERMINATION\\n"
                "   - How long does this last?\\n"
                "   - Does it auto-renew?\\n"
                "   - How can each party get out of it?\\n"
                "   - What notice is required?\\n"
                "   - What happens after termination?\\n\\n"
                "6. COMPLIANCE & LEGAL OBLIGATIONS\\n"
                "   - Any laws, regulations, or licenses mentioned\\n"
                "   - Data protection, privacy requirements\\n"
                "   - Insurance, bonding, security requirements\\n"
                "   - Audit rights, reporting obligations\\n\\n"
                "7. IMPORTANT DEADLINES\\n"
                "   - Payment due dates\\n"
                "   - Delivery schedules\\n"
                "   - Reporting timelines\\n"
                "   - Review or renewal dates\\n\\n"
                "8. ATTACHMENTS/SCHEDULES\\n"
                "   - List any annexures, exhibits, SOWs, schedules mentioned\\n\\n"
                "9. COMPLEX LEGAL TERMS\\n"
                "   - Identify 5-8 legal terms that a non-lawyer might not understand\\n"
                "   - Provide simple, everyday language explanations\\n"
                "   - Examples: indemnification, force majeure, severability, liquidated damages, etc.\\n\\n"
                "10. EXECUTIVE SUMMARY\\n"
                "   - Write 2-3 paragraphs in plain language\\n"
                "   - Should be understandable by someone with no legal training\\n"
                "   - Cover: what this is, who's involved, what happens, key numbers, how long it lasts\\n"
                "   - Use analogies or everyday examples if helpful\\n"
                "   - 150-250 words\\n\\n"
                "REMEMBER: Your goal is to make this legal document fully understandable to a non-lawyer "
                "while capturing all essential information accurately."
            )
        ])
        
        # Configure LLM with conservative settings to reduce None returns
        import time
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Use gemini-2.5-flash for comprehensive summary
        model_for_summary = _get_llm_model_name()  # Consistent with main config
        
        # Fallback to base model if primary fails
        fallback_models = [
            "gemini-1.5-flash"  # Fallback option
        ]
        
        logger.info(f"Using {model_for_summary} for comprehensive summary (optimized for quota efficiency)")
        
        # Use a dedicated LLM instance with optimized settings for structured output
        # CRITICAL: Do NOT use response_mime_type with with_structured_output() - they conflict!
        summary_llm = ChatGoogleGenerativeAI(
            model=model_for_summary,
            temperature=0.2,  # Slightly higher for Flash (better quality)
            max_output_tokens=2048,  # Standard limit for most models
            google_api_key=settings.GEMINI_API_KEY,
            max_retries=1,  # Reduced retries to avoid quota waste
            request_timeout=60,  # Standard timeout
            # DO NOT set response_mime_type - it conflicts with with_structured_output()
            # DO NOT set candidate_count=1 - causes empty responses with structured output
        )
        
        structured_summary_llm = summary_llm.with_structured_output(ComprehensiveSummary)
        chain = summary_prompt | structured_summary_llm
        
        # Limit document text to avoid token limits (use first 6000 chars for more reliable processing)
        truncated_text = full_text[:6000]  # Reduced from 8000 for better reliability
        if len(full_text) > 6000:
            # Add context about truncation
            truncated_text += "\\n\\n[Document continues for " + str(len(full_text) - 6000) + " more characters...]\\n\\nNote: Analyze the provided excerpt and extract all available information."
        
        logger.info(f"Generating comprehensive {doc_type_name} summary using LLM (text length: {len(truncated_text)} chars)...")
        
        # Invoke with retry logic and model fallback
        result = None
        max_attempts = 2  # Reduced to 2 since we have model fallback
        last_error = None
        
        # Try primary model, then fallback models
        models_to_try = [model_for_summary] + fallback_models
        
        for model_name in models_to_try:
            logger.info(f"Trying model: {model_name}")
            
            # Create LLM instance for this model
            try:
                # Adjust parameters based on model type
                is_pro = "pro" in model_name
                is_exp = "exp" in model_name
                
                current_llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0.2,  # Consistent across models
                    max_output_tokens=2048,  # Standard limit
                    google_api_key=settings.GEMINI_API_KEY,
                    max_retries=1,  # Single retry to save quota
                    request_timeout=75 if is_pro else 60,  # Pro gets slightly more time
                )
                
                current_structured_llm = current_llm.with_structured_output(ComprehensiveSummary)
                current_chain = summary_prompt | current_structured_llm
                
            except Exception as model_init_error:
                logger.warning(f"Failed to initialize {model_name}: {model_init_error}")
                last_error = model_init_error
                continue
            for attempt in range(max_attempts):
                try:
                    logger.info(f"Model {model_name}, attempt {attempt + 1}/{max_attempts}...")
                    
                    result = current_chain.invoke(
                        {'document_text': truncated_text},
                        config={"max_retries": 0, "request_timeout": 60}  # No retries here, we handle it ourselves
                    )
                    
                    # Check immediately if result is None
                    if result is None:
                        logger.warning(f"{model_name} returned None on attempt {attempt + 1}")
                        if attempt < max_attempts - 1:
                            wait_time = 2  # Fixed 2 second wait
                            logger.info(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.warning(f"{model_name} returned None after {max_attempts} attempts, trying next model")
                            last_error = ValueError(f"{model_name} returned None")
                            break  # Try next model
                    
                    # Validate result has expected structure
                    if not hasattr(result, 'model_dump') and not hasattr(result, 'dict') and not isinstance(result, dict):
                        logger.warning(f"{model_name} returned unexpected type: {type(result)}")
                        if attempt < max_attempts - 1:
                            time.sleep(2)
                            continue
                        else:
                            logger.warning(f"{model_name} returned wrong type after {max_attempts} attempts, trying next model")
                            last_error = ValueError(f"Unexpected type: {type(result)}")
                            break  # Try next model
                    
                    # Success - we got a valid result!
                    logger.info(f"✅ {model_name} returned valid result on attempt {attempt + 1}")
                    break  # Break attempt loop
                    
                except Exception as invoke_exc:
                    error_msg = str(invoke_exc)
                    logger.warning(f"{model_name} attempt {attempt + 1} failed: {error_msg}")
                    last_error = invoke_exc
                    
                    # Check for quota/rate limit errors
                    is_quota = any(indicator in error_msg.lower() for indicator in 
                                  ['quota', 'rate limit', '429', 'resource exhausted', 'quota exceeded'])
                    
                    if is_quota:
                        logger.warning(f"Quota/rate limit detected: {error_msg}")
                        raise  # Don't retry on quota issues, trigger fallback immediately
                    
                    # For other errors, retry if attempts remain
                    if attempt < max_attempts - 1:
                        wait_time = 2
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.warning(f"{model_name} failed after {max_attempts} attempts, trying next model")
                        break  # Try next model
            
            # Check if we got a valid result from this model
            if result is not None:
                logger.info(f"✅ Successfully got result from {model_name}")
                break  # Break model loop - we're done!
        
        # After trying all models, check if we got a result
        if result is None:
            error_message = f"All models failed to generate output. Last error: {last_error}"
            logger.error(error_message)
            raise ValueError(error_message)
        
        logger.info(f"✅ LLM returned result type: {type(result)}")
        
        # Final validation that result is not None
        if result is None:
            logger.error("LLM returned None after all retry attempts - falling back to regex extraction")
            raise ValueError("LLM returned None after retry logic - likely API quota or persistent connection issue")
        
        # Extract data from result with comprehensive error handling
        try:
            if hasattr(result, 'model_dump'):
                summary_dict = result.model_dump()
                logger.info("Extracted data using model_dump()")
            elif hasattr(result, 'dict'):
                summary_dict = result.dict()
                logger.info("Extracted data using dict()")
            elif isinstance(result, dict):
                summary_dict = result
                logger.info("Result is already a dict")
            else:
                # Last resort - try to convert to dict
                summary_dict = dict(result) if result is not None else {}
                logger.warning(f"Converted result to dict using dict() constructor")
                
            # Validate we got a non-empty dictionary
            if not summary_dict:
                logger.warning("Extracted summary_dict is empty")
                raise ValueError("Extracted empty dictionary from LLM result")
                
        except (TypeError, ValueError, AttributeError) as extract_error:
            logger.error(f"Failed to extract data from LLM result: {extract_error}", exc_info=True)
            raise ValueError(f"Failed to extract data from LLM result: {extract_error}")
        
        # Ensure all required fields exist with defaults
        summary_dict.setdefault('document_type', doc_type_name)
        summary_dict.setdefault('parties', [])
        summary_dict.setdefault('purpose', 'Purpose not extracted')
        summary_dict.setdefault('key_obligations', {})
        summary_dict.setdefault('financial_terms', [])
        summary_dict.setdefault('legal_terms_explained', [])
        summary_dict.setdefault('compliance_requirements', [])
        summary_dict.setdefault('important_deadlines', [])
        summary_dict.setdefault('attachments_mentioned', [])
        
        if not summary_dict.get('term_and_termination'):
            summary_dict['term_and_termination'] = {
                'duration': 'Not specified',
                'termination_process': 'Not specified',
                'simple_explanation': 'Unable to extract termination details'
            }
        
        if not summary_dict.get('executive_summary'):
            summary_dict['executive_summary'] = textwrap.shorten(full_text, width=250, placeholder='...')
        
        logger.info(f"✅ Comprehensive summary generated: {len(summary_dict.get('executive_summary', ''))} chars, "
                   f"{len(summary_dict.get('parties', []))} parties, "
                   f"{len(summary_dict.get('legal_terms_explained', []))} terms explained")
        
        return summary_dict
        
    except Exception as exc:
        error_msg = str(exc)
        # Check if this is a quota/rate limit issue
        is_quota_issue = any(indicator in error_msg.lower() for indicator in 
                            ['quota', 'rate limit', '429', 'resource exhausted', 'quota exceeded'])
        
        if is_quota_issue:
            logger.warning(f"LLM quota exceeded for comprehensive summary, using regex fallback: {error_msg}")
        else:
            logger.error(f"Comprehensive summary LLM generation failed: {exc}", exc_info=True)
        
        # Fallback to regex-based extraction (more intelligent than basic)
        logger.info("Falling back to regex-based comprehensive summary extraction")
        try:
            return _generate_comprehensive_summary_from_analysis(
                full_text=full_text,
                doc_type=doc_type,
                doc_type_name=doc_type_name,
                chunk_results=[],
                deduped_clauses=[]
            )
        except Exception as fallback_exc:
            logger.error(f"Regex fallback also failed: {fallback_exc}", exc_info=True)
            # Final fallback to minimal summary
            return {
                'document_type': doc_type_name,
                'executive_summary': textwrap.shorten(full_text, width=500, placeholder='...'),
                'parties': [],
                'purpose': 'Unable to generate detailed analysis. Please review document manually.',
                'key_obligations': {},
                'financial_terms': [],
                'term_and_termination': {
                    'duration': 'Not specified',
                    'termination_process': 'Not specified',
                    'simple_explanation': 'Unable to extract termination details'
                },
                'legal_terms_explained': [],
            }
def extract_text_from_file(uploaded_file):
    """Extract text depending on file type."""
    if uploaded_file.name.endswith('.pdf'):
        try:
            text = ""
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except fitz.FileDataError as e:
            return None
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    elif uploaded_file.name.endswith('.docx'):
        doc = Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs])

    elif uploaded_file.name.endswith('.txt'):
        return uploaded_file.read().decode('utf-8')

    else:
        return None

@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def summarize_document(request):
    """API endpoint for document summarization with async processing support"""
    try:
        # Check if async mode is requested
        async_value = request.data.get('async', 'false')
        async_mode = str(async_value).lower() == 'true' if async_value else False
        
        uploaded_file = request.FILES.get('document')
        if not uploaded_file:
            return Response({
                'error': 'Please upload a document'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Check file size (limit to 10MB)
        if uploaded_file.size > 10 * 1024 * 1024:
            return Response({
                'error': 'File size exceeds 10MB limit.'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Reset file pointer to beginning for reading content
        uploaded_file.seek(0)

        try:
            text = extract_text_from_file(uploaded_file)
            if text is None:
                return Response({
                    'error': 'Error extracting text from file.'
                }, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({
                'error': f'Error extracting text from file: {str(e)}'
            }, status=status.HTTP_400_BAD_REQUEST)
            
        if not text:
            return Response({
                'error': 'Unsupported file type. Please upload PDF, DOCX, or TXT'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Get user from JWT token
        user = request.user
        
        if not user:
            return Response({
                'error': 'User not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # Create session first
        session = DocumentSession(
            user=user,
            document_text=text,
            summary='Processing...',  # Placeholder
            highlighted_preview='',
            high_risk_clauses=[],
        )
        session.save()
        
        # If async mode, queue the task and return immediately
        if async_mode:
            from .tasks import analyze_document_async
            
            # Set initial task status
            set_task_status(str(session.id), 'pending', 0, 'Queued for analysis')
            
            # Queue the async task
            analyze_document_async.delay(str(session.id), text)
            
            return Response({
                'success': True,
                'async': True,
                'session_id': str(session.id),
                'filename': uploaded_file.name,
                'status': 'pending',
                'message': 'Document queued for analysis. Check status endpoint for progress.'
            }, status=status.HTTP_202_ACCEPTED)
        
        # Synchronous processing (original behavior)
        analysis = generate_document_analysis(text)
        # Update session with analysis results
        session.summary = analysis.get('summary') or textwrap.shorten(
            text.replace('\n', ' '),
            width=500,
            placeholder='…'
        )
        session.highlighted_preview = analysis.get('highlighted_preview') or html.escape(text).replace('\n', '<br />')
        session.high_risk_clauses = analysis.get('high_risk_clauses') or []
        session.comprehensive_summary = analysis.get('comprehensive_summary')
        session.document_type = analysis.get('document_type')
        session.document_type_confidence = analysis.get('document_type_confidence')
        session.save()
        
        preview_text = analysis.get('preview_text') or text
        
        return Response({
            'success': True,
            'async': False,
            'summary': session.summary,
            'comprehensive_summary': analysis.get('comprehensive_summary'),  # ADD THIS
            'highlighted_preview': session.highlighted_preview,
            'high_risk_clauses': session.high_risk_clauses,
            'preview_text': preview_text,
            'document_text': text,
            'document_type': analysis.get('document_type'),  # ADD THIS TOO
            'document_type_confidence': analysis.get('document_type_confidence'),  # AND THIS
            'session_id': str(session.id),
            'filename': uploaded_file.name
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

  from django.views.decorators.csrf import csrf_exempt # Added csrf_exempt import

# ...

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def chat_message(request):
    """Handle chat messages"""
    try:
        user_message = request.data.get('message')
        session_id = request.data.get('session_id')
        
        if not user_message or not session_id:
            return Response({
                'error': 'Missing message or session_id'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get user from JWT token (request.user is already a User object)
        user = request.user
        
        if not user:
            return Response({
                'error': 'User not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # Get the document session and verify ownership
        try:
            session = DocumentSession.objects(id=session_id).first()
            if not session:
                return Response({
                    'error': 'Session not found'
                }, status=status.HTTP_404_NOT_FOUND)
                
            if str(session.user.id) != str(user.id):
                return Response({
                    'error': 'Access denied'
                }, status=status.HTTP_403_FORBIDDEN)
        except DoesNotExist:
            return Response({
                'error': 'Session not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # Save user message
        user_msg = ChatMessage(
            session=session,
            message=user_message,
            is_user=True
        )
        user_msg.save()
        
        # Get AI response
        ai_response = chat_with_document(session, user_message)
        
        # Save AI response
        ai_msg = ChatMessage(
            session=session,
            message=ai_response,
            is_user=False
        )
        ai_msg.save()
        
        return Response({
            'response': ai_response,
            'message_id': str(ai_msg.id)
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def chat_history(request, session_id):
    """Get chat history for a session"""
    try:
        # Get user from JWT token (request.user is already a User object)
        user = request.user
        
        if not user:
            return Response({
                'error': 'User not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        try:
            session = DocumentSession.objects(id=session_id).first()
            if not session:
                return Response({
                    'error': 'Session not found'
                }, status=status.HTTP_404_NOT_FOUND)
                
            if str(session.user.id) != str(user.id):
                return Response({
                    'error': 'Access denied'
                }, status=status.HTTP_403_FORBIDDEN)
        except DoesNotExist:
            return Response({
                'error': 'Session not found'
            }, status=status.HTTP_404_NOT_FOUND)
            
        messages = ChatMessage.objects(session=session).order_by('created_at')
        
        messages_data = [
            {
                'id': str(msg.id),
                'message': msg.message,
                'is_user': msg.is_user,
                'timestamp': msg.created_at.isoformat()
            }
            for msg in messages
        ]
        
        return Response({
            'messages': messages_data,
            'session': {
                'id': str(session.id),
                'summary': session.summary,
                'highlighted_preview': session.highlighted_preview or '',
                'high_risk_clauses': session.high_risk_clauses or [],
                'preview_text': session.document_text,
                'comprehensive_summary': session.comprehensive_summary or None,
                'document_type': session.document_type or None,
                'document_type_confidence': session.document_type_confidence or None,
                'created_at': session.created_at.isoformat()
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_sessions(request):
    """Get user's document sessions"""
    try:
        user = request.user
        
        if not user:
            logger.error("User not found in request")
            return Response({
                'error': 'User not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        logger.info(f"Fetching sessions for user: {user.email if hasattr(user, 'email') else user}")
        
        try:
            # Query sessions for this user
            sessions = list(DocumentSession.objects(user=user).order_by('-created_at'))
            logger.info(f"Found {len(sessions)} sessions for user {user.email if hasattr(user, 'email') else user}")
        except Exception as query_error:
            logger.error(f"Error querying DocumentSession: {str(query_error)}")
            return Response({
                'error': f'Database query failed: {str(query_error)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        if not sessions:
            # No sessions found, return empty list
            return Response({
                'sessions': []
            }, status=status.HTTP_200_OK)
        
        session_ids = [session.id for session in sessions]
        
        try:
            # Aggregate message counts for all sessions in a single query
            message_counts = list(ChatMessage._collection.aggregate([
                {'$match': {'session': {'$in': session_ids}}},
                {'$group': {'_id': '$session', 'count': {'$sum': 1}}}
            ]))
            
            # Convert list of dicts to a dict for easy lookup
            message_counts_map = {item['_id']: item['count'] for item in message_counts}
        except Exception as agg_error:
            logger.error(f"Error aggregating message counts: { str(agg_error)}")
            # Continue without message counts
            message_counts_map = {}
        
        sessions_data = []
        for session in sessions:
            try:
                message_count = message_counts_map.get(session.id, 0)
                sessions_data.append({
                    'id': str(session.id),
                    'summary_preview': (session.summary[:150] + '...' if len(session.summary) > 150 else session.summary) if session.summary else '',
                    'created_at': session.created_at.isoformat() if session.created_at else None,
                    'message_count': message_count,
                    'document_preview': (session.document_text[:100] + '...' if len(session.document_text) > 100 else session.document_text) if session.document_text else '',
                    'highlighted_preview': session.highlighted_preview or '',
                    'high_risk_clause_count': len(session.high_risk_clauses or []),
                    'comprehensive_summary': session.comprehensive_summary or None,
                    'document_type': session.document_type or None,
                    'document_type_confidence': session.document_type_confidence or None,
                })
            except Exception as session_error:
                logger.error(f"Error processing session {session.id}: {str(session_error)}")
                # Skip this session and continue
                continue
        
        logger.info(f"Successfully prepared {len(sessions_data)} sessions data")
        return Response({
            'sessions': sessions_data
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Unexpected error in user_sessions: {str(e)}", exc_info=True)
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def task_status(request, session_id):
    """Get the status of an async document analysis task"""
    try:
        user = request.user
        
        if not user:
            return Response({
                'error': 'User not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # Verify session ownership
        try:
            session = DocumentSession.objects(id=session_id).first()
            if not session:
                return Response({
                    'error': 'Session not found'
                }, status=status.HTTP_404_NOT_FOUND)
                
            if str(session.user.id) != str(user.id):
                return Response({
                    'error': 'Access denied'
                }, status=status.HTTP_403_FORBIDDEN)
        except DoesNotExist:
            return Response({
                'error': 'Session not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # Get task status from cache
        status_info = get_task_status(session_id)
        
        if not status_info:
            # No status in cache - check if session has results
            if session.summary and session.summary != 'Processing...':
                status_info = {
                    'status': 'completed',
                    'progress': 100,
                    'message': 'Analysis complete'
                }
            else:
                status_info = {
                    'status': 'unknown',
                    'progress': 0,
                    'message': 'No status information available'
                }
        
        # If completed, include results
        response_data = {
            'session_id': str(session.id),
            **status_info
        }
        
        if status_info['status'] == 'completed':
            response_data.update({
                'summary': session.summary,
                'highlighted_preview': session.highlighted_preview or '',
                'high_risk_clauses': session.high_risk_clauses or [],
                'high_risk_clause_count': len(session.high_risk_clauses or [])
            })
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def session_detail(request, session_id):
    """View a specific session with chat"""
    try:
        # Get user from JWT token (request.user is already a User object)
        user = request.user
        
        if not user:
            return Response({
                'error': 'User not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        try:
            session = DocumentSession.objects(id=session_id).first()
            if not session:
                return Response({
                    'error': 'Session not found'
                }, status=status.HTTP_404_NOT_FOUND)
                
            if str(session.user.id) != str(user.id):
                return Response({
                    'error': 'Access denied'
                }, status=status.HTTP_403_FORBIDDEN)
        except DoesNotExist:
            return Response({
                'error': 'Session not found'
            }, status=status.HTTP_404_NOT_FOUND)
            
        chat_messages = ChatMessage.objects(session=session).order_by('created_at')
        
        messages_data = [
            {
                'id': str(msg.id),
                'message': msg.message,
                'is_user': msg.is_user,
                'timestamp': msg.created_at.isoformat()
            }
            for msg in chat_messages
        ]
        
        return Response({
            'session': {
                'id': str(session.id),
                'summary': session.summary,
                'highlighted_preview': session.highlighted_preview or '',
                'high_risk_clauses': session.high_risk_clauses or [],
                'preview_text': session.document_text,
                'document_text': session.document_text,
                'created_at': session.created_at.isoformat()
            },
            'messages': messages_data
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
