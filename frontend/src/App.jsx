import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import MissionList from './pages/MissionList'
import MissionView from './pages/MissionView'

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-dt-bg">
        {/* Header */}
        <header className="fixed top-0 left-0 right-0 z-50 glass-strong">
          <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <a href="/missions" className="flex items-center gap-3 group">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-dt-accent to-dt-success flex items-center justify-center">
                <svg className="w-6 h-6 text-dt-bg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <div>
                <h1 className="font-display text-xl font-bold text-gradient">DeepThinker</h1>
                <p className="text-xs text-dt-text-dim font-mono">v2.0</p>
              </div>
            </a>
            
            <nav className="flex items-center gap-6">
              <a href="/missions" 
                className="text-sm text-dt-text-dim hover:text-dt-accent transition-colors">
                Missions
              </a>
              <a href="/api/docs" target="_blank"
                className="text-sm text-dt-text-dim hover:text-dt-accent transition-colors">
                API Docs
              </a>
            </nav>
          </div>
        </header>

        {/* Main content */}
        <main className="pt-20">
          <Routes>
            <Route path="/" element={<Navigate to="/missions" replace />} />
            <Route path="/missions" element={<MissionList />} />
            <Route path="/missions/:id" element={<MissionView />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App

