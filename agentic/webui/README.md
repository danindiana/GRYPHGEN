# GRYPHGEN Agentic Web Dashboard

Modern React-based web interface for GRYPHGEN Agentic AI development assistant.

## Features

- **Real-time Code Generation**: Interactive code generation with live preview
- **Project Management**: Visual project and task management dashboard
- **Analytics**: Performance metrics and usage analytics
- **Collaboration**: Real-time collaboration via WebSockets
- **Dark Mode**: Beautiful dark theme optimized for developers

## Tech Stack

- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **State Management**: Zustand
- **Data Fetching**: TanStack Query (React Query)
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Forms**: React Hook Form + Zod validation
- **Icons**: Lucide React

## Quick Start

### Install Dependencies

```bash
cd webui
npm install
```

### Development Server

```bash
npm run dev
```

Open http://localhost:5173

### Build for Production

```bash
npm run build
```

Output will be in `dist/` directory.

## Project Structure

```
webui/
├── public/             # Static assets
├── src/
│   ├── components/    # Reusable React components
│   ├── pages/         # Page components
│   ├── services/      # API services
│   ├── hooks/         # Custom React hooks
│   ├── styles/        # Global styles
│   ├── types/         # TypeScript type definitions
│   ├── utils/         # Utility functions
│   ├── App.tsx        # Main app component
│   └── main.tsx       # Entry point
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.js
```

## Configuration

### API Endpoint

Update API base URL in `src/services/api.ts`:

```typescript
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
```

Set in `.env`:

```
VITE_API_URL=https://api.gryphgen.ai
```

### WebSocket Connection

Configure WebSocket URL in `src/services/websocket.ts`.

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript type checking

## Environment Variables

Create `.env` file:

```bash
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
```

## License

MIT
