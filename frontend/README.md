# Advoc AI Frontend

Modern React + Vite interface for the Advoc AI platform. It showcases AI-assisted legal tooling — document analysis, smart drafting, secure e-signatures, and lawyer collaboration — wrapped in a highly polished marketing site with interactive demos and authentication-aware CTAs.

## Features

- **AI document workflow**: Analyzer, creator, timeline, and e-signature sections with modal detail views.
- **Auth-aware CTAs**: Buttons automatically route to the correct experience based on login state handled via `AuthContext`.
- **Reusable UI system**: Radix primitives, Tailwind utility classes, and custom components (navbar, sidebar, modals, comments).
- **3D/visual polish**: Subtle particle background (`Subtle3DBackground`), gradients, and floating UI cards powered by `@react-three/fiber` + `tsparticles`.
- **API integration ready**: Centralized Axios instance with token refresh logic (`src/api/axios.js`) that respects public vs. protected routes.

## Tech Stack

- React 19, React Router 7, Vite 7
- Tailwind CSS 3 + tailwind-merge + tailwindcss-animate
- Radix UI, Lucide icons, TipTap editor suite
- Three.js ecosystem (`@react-three/fiber`, `@react-three/drei`)
- Axios for HTTP, React Hot Toast for notifications

## Getting Started

### Prerequisites

- Node.js 18+ and npm 9+ (recommend the latest LTS)

### Installation

```bash
npm install
```

### Local Development

```bash
npm run dev
```

Vite serves the app on `http://localhost:5173` by default. The dev server supports hot module reloading and automatically proxies environment variables prefixed with `VITE_`.

### Build & Preview

```bash
npm run build   # production build in dist/
npm run preview # serves the production bundle locally
```

### Lint

```bash
npm run lint
```

## Environment Variables

Create a `.env` (or `.env.local`) at the project root:

```
VITE_API_BASE_URL=https://your-backend.example.com/
```

`src/api/axios.js` falls back to `http://localhost:8000/` when the variable is not provided. Set it to your API gateway so the auth interceptors can refresh and append tokens correctly.

## Project Structure

```
src/
├─ App.jsx                  # Route definitions
├─ main.jsx                 # React entry + providers
├─ pages/                   # Page-level views (Home, Chat, Lawyers, Auth, etc.)
├─ Components/
│  ├─ Navbar/               # Auth-aware navigation
│  ├─ Footer.jsx            # Shared footer assets
│  ├─ Modals/               # Signature & sharing flows
│  ├─ ui/                   # Button, Card, Input, Tabs wrappers
│  └─ Subtle3DBackground.jsx
├─ api/axios.js             # Axios instance with interceptors
├─ context/AuthContext.jsx  # Auth state + helpers
├─ styles/                  # Additional CSS (Markdown preview)
└─ utils/                   # Helpers and API utilities
```

## Conventions

- Components live in `PascalCase` folders/files, hooks/utilities in `camelCase`.
- Keep shared styling inside Tailwind classes; avoid ad-hoc CSS unless necessary.
- Use the provided UI components (`src/Components/ui`) for consistent spacing, typography, and variants.
- Prefer the centralized Axios instance and `AuthContext` when adding network or auth-aware logic.

## Next Steps

- Wire up real API endpoints in pages such as `Chat`, `DocumentAnalyzer`, and `LawyerDashboard`.
- Flesh out the footer or navigation links to route to actual product pages.
- Add tests or storybook examples for critical UI primitives if needed.

Happy building! 

