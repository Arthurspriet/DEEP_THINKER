/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  // Enable dynamic class generation for color utilities
  safelist: [
    // Dynamic color classes used by panels
    {
      pattern: /^(bg|text|border)-(emerald|amber|yellow|red|cyan|violet|indigo|rose|teal|orange|slate)-(400|500)(\/\d+)?$/,
    },
    {
      pattern: /^from-(emerald|amber|yellow|red|cyan|violet|indigo|rose|teal|orange)-(400|500)$/,
    },
    {
      pattern: /^to-(emerald|amber|yellow|red|cyan|violet|indigo|rose|teal|orange)-(400|500)$/,
    },
    {
      pattern: /^shadow-(emerald|amber|yellow|red|cyan|violet|indigo|rose|teal|orange)-(500)(\/\d+)?$/,
    },
  ],
  theme: {
    extend: {
      colors: {
        'dt': {
          // Core background shades (deep navy/charcoal)
          'bg': '#0a0c10',
          'bg-elevated': '#0f1218',
          'surface': '#141920',
          'surface-light': '#1c222d',
          'surface-lighter': '#252d3a',
          // Borders
          'border': '#2a3344',
          'border-bright': '#3d4a5f',
          // Accent (cyber cyan)
          'accent': '#00d4ff',
          'accent-dim': '#0099b8',
          'accent-bright': '#4de8ff',
          // Status colors
          'success': '#00ff88',
          'success-dim': '#00cc6a',
          'warning': '#ffb800',
          'warning-dim': '#cc9300',
          'error': '#ff4466',
          'error-dim': '#cc3652',
          // Text
          'text': '#e8ecf4',
          'text-secondary': '#a8b4c8',
          'text-dim': '#6b7a94',
          'text-muted': '#4a5568',
          // Panel-specific accents
          'governance': '#f59e0b',  // Amber
          'research': '#06b6d4',    // Cyan
          'ml': '#8b5cf6',          // Violet
          'supervisor': '#6366f1',  // Indigo
          'analysis': '#f43f5e',    // Rose
          'synthesis': '#14b8a6',   // Teal
          'epistemic': '#f97316',   // Orange
        }
      },
      fontFamily: {
        // Display: geometric, futuristic
        'display': ['Orbitron', 'SF Pro Display', 'system-ui', 'sans-serif'],
        // Mono: for data, metrics, code
        'mono': ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
        // Sans: clean, readable body text
        'sans': ['Geist', 'Inter', 'system-ui', 'sans-serif'],
        // Heading: distinctive headers
        'heading': ['Space Grotesk', 'SF Pro Text', 'system-ui', 'sans-serif'],
      },
      fontSize: {
        '2xs': ['0.625rem', { lineHeight: '0.75rem' }],
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'flow': 'flow 3s linear infinite',
        'fade-in': 'fade-in 0.3s ease-out',
        'fade-in-up': 'fade-in-up 0.4s ease-out',
        'slide-up': 'slide-up 0.4s ease-out',
        'slide-in-right': 'slide-in-right 0.3s ease-out',
        'scale-in': 'scale-in 0.2s ease-out',
        'shimmer': 'shimmer 2s linear infinite',
        'gradient-shift': 'gradient-shift 3s ease infinite',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { 
            boxShadow: '0 0 20px rgba(0, 212, 255, 0.3), inset 0 0 20px rgba(0, 212, 255, 0.05)'
          },
          '50%': { 
            boxShadow: '0 0 40px rgba(0, 212, 255, 0.6), inset 0 0 30px rgba(0, 212, 255, 0.1)'
          },
        },
        'flow': {
          '0%': { backgroundPosition: '200% 0' },
          '100%': { backgroundPosition: '-200% 0' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'fade-in-up': {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-up': {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-in-right': {
          '0%': { opacity: '0', transform: 'translateX(20px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        'scale-in': {
          '0%': { opacity: '0', transform: 'scale(0.95)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        'shimmer': {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        'gradient-shift': {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
      },
      backdropBlur: {
        'xs': '2px',
        '3xl': '64px',
      },
      boxShadow: {
        'glow': '0 0 20px rgba(0, 212, 255, 0.3)',
        'glow-lg': '0 0 40px rgba(0, 212, 255, 0.4)',
        'inner-glow': 'inset 0 0 20px rgba(0, 212, 255, 0.1)',
      },
      backgroundImage: {
        'grid-pattern': `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%232a3344' fill-opacity='0.3'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
        'noise': `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
      },
    },
  },
  plugins: [],
}
