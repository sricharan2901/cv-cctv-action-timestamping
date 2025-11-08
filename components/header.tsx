export default function Header() {
  return (
    <header className="bg-card border-b border-border sticky top-0 z-50 backdrop-blur-md bg-card/95">
      <div className="container mx-auto px-4 py-4 md:py-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary/20 rounded-lg">
            <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
              />
            </svg>
          </div>
          <div>
            <h1 className="text-2xl md:text-3xl font-bold text-balance">CCTV Action Recognition</h1>
            <p className="text-sm text-muted-foreground">Advanced AI-powered action detection system</p>
          </div>
        </div>
      </div>
    </header>
  )
}
