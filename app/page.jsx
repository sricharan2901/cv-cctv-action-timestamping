"use client"

import { useState, useRef } from "react"

export default function Home() {
  // Video upload state
  const [videoFile, setVideoFile] = useState(null)
  const [videoPreview, setVideoPreview] = useState("")
  const [videoLoading, setVideoLoading] = useState(false)
  const [videoResults, setVideoResults] = useState(null)
  const [searchQuery, setSearchQuery] = useState("")
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchResults, setSearchResults] = useState(null)
  const fileInputRef = useRef(null)
  const videoRef = useRef(null)

  // Video handling
  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      setVideoFile(selectedFile)
      const previewUrl = URL.createObjectURL(selectedFile)
      setVideoPreview(previewUrl)
      setVideoResults(null)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    const droppedFile = e.dataTransfer.files?.[0]
    if (droppedFile && droppedFile.type.startsWith("video/")) {
      setVideoFile(droppedFile)
      const previewUrl = URL.createObjectURL(droppedFile)
      setVideoPreview(previewUrl)
      setVideoResults(null)
    }
  }

  const handleUpload = async () => {
    if (!videoFile) return

    setVideoLoading(true)
    try {
      const formData = new FormData()
      formData.append("video", videoFile)

      const response = await fetch("/api/analyze-video", {
        method: "POST",
        body: formData,
      })

      const data = await response.json()
      setVideoResults(data)
    } catch (error) {
      console.error("Upload error:", error)
      setVideoResults({
        error: "Failed to analyze video",
        success: false,
      })
    } finally {
      setVideoLoading(false)
    }
  }

  // Search handling
  const handleSearch = async () => {
    if (!searchQuery.trim()) return

    setSearchLoading(true)
    try {
      const response = await fetch("/api/search-action", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: searchQuery.trim() }),
      })

      const data = await response.json()
      setSearchResults(data)
    } catch (error) {
      console.error("Search error:", error)
      setSearchResults({
        error: "Failed to search",
        success: false,
      })
    } finally {
      setSearchLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      handleSearch()
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-[#000022] via-[#000044] to-[#555555] relative text-foreground">
      <section className="container mx-auto px-6 py-16 md:py-24 text-center">
        <h2 className="text-5xl md:text-7xl font-bold tracking-tight mb-6 text-balance leading-tight">
          <span className="bg-gradient-to-r from-accent via-primary to-blue-400 bg-clip-text text-transparent">
            Advanced CCTV
          </span>
          <br />
          <span className="bg-gradient-to-r from-accent via-primary to-blue-400 bg-clip-text text-transparent">
            Video Captioning
          </span>
        </h2>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-8">
          Powered by AI to detect and find the exact moment of the actions in real-time surveillance footage
        </p>
        <div className="w-1 h-1 bg-gradient-to-r from-accent to-primary rounded-full mx-auto mb-8"></div>
      </section>

      <div className="container mx-auto px-6 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          {/* Upload Section */}
          <div className="space-y-6">
            <div>
              <h3 className="text-3xl font-bold mb-2">Upload & Analyze</h3>
              <p className="text-muted-foreground">Drop the footage to pass it for captioning</p>
            </div>

            <div
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              className="relative group cursor-pointer"
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-accent/20 to-primary/20 rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              <div className="relative border-2 border-dashed border-border/40 border-white/20 rounded-2xl p-12 hover:border-accent/50 transition-all duration-300 bg-transparent">
                {videoPreview ? (
                  <div className="space-y-4">
                    <div className="relative rounded-xl overflow-hidden bg-black/60">
                      <video src={videoPreview} className="w-full h-48 object-cover" />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent" />
                    </div>
                    <p className="text-sm text-muted-foreground truncate text-center">{videoFile?.name}</p>
                  </div>
                ) : (
                  <div className="text-center space-y-4">
                    <div className="w-16 h-16 rounded-full bg-gradient-to-br from-accent/20 to-primary/20 flex items-center justify-center mx-auto">
                      <svg className="w-8 h-8 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={1.5}
                          d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                        />
                      </svg>
                    </div>
                    <div>
                      <p className="text-lg font-semibold">Drop video here</p>
                      <p className="text-sm text-muted-foreground">or click to browse</p>
                    </div>
                  </div>
                )}
                <input ref={fileInputRef} type="file" accept="video/*" onChange={handleFileChange} className="hidden" />
              </div>
            </div>

            <button
              onClick={handleUpload}
              disabled={!videoFile || videoLoading}
              className="w-full px-8 py-4 rounded-xl font-semibold text-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed bg-white/20 hover:shadow-lg hover:shadow-accent/30 text-white"
            >
              {videoLoading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-5 h-5 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Analyzing Video...
                </span>
              ) : (
                "Caption Video"
              )}
            </button>
          </div>

          <div className="space-y-6">
            <div>
              <h3 className="text-3xl font-bold mb-2">Search Actions</h3>
              <p className="text-muted-foreground">Find specific actions in your footage</p>
            </div>

            <div className="space-y-4">
              <div className="relative group">
                <div className="absolute inset-0 bg-white/20 rounded-xl blur-lg opacity-0 group-focus-within:opacity-100 transition-opacity duration-300"></div>
                <input
                  type="text"
                  placeholder="Search for actions: running, falling, fighting..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={handleKeyPress}
                  className="relative w-full px-6 py-4 rounded-xl bg-card bg-white/5 border border-white/5 border-border/40 focus:border-primary/50 outline-none transition-all duration-300 placeholder:text-muted-foreground text-lg"
                />
              </div>

              <button
                onClick={handleSearch}
                disabled={!searchQuery.trim() || searchLoading}
                className="w-full px-8 py-4 rounded-xl font-semibold text-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed bg-white/20 hover:shadow-lg hover:shadow-primary/30 text-white"
              >
                {searchLoading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="w-5 h-5 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                      />
                    </svg>
                    Searching...
                  </span>
                ) : (
                  "Search"
                )}
              </button>

              <div className="p-4 rounded-xl bg-card/50 bg-transparent ">
                <p className="text-sm text-muted-foreground">
                  <span className="font-semibold text-foreground block mb-2">Supported Actions:</span>
                  Running, walking, falling, fighting, jumping, sitting, standing, climbing
                </p>
              </div>
            </div>
          </div>
        </div>

        {videoResults && !videoResults.error && (
          <div className="space-y-12">
            <div className="border-t border-border/20 pt-12">
              <h2 className="text-4xl font-bold mb-8">Analysis Results</h2>

              <div className="bg-black rounded-2xl overflow-hidden shadow-2xl mb-8">
                <div className="relative">
                  <video ref={videoRef} src={videoPreview} controls className="w-full aspect-video bg-black" />
                  <div className="absolute top-4 right-4 bg-black/70 backdrop-blur-md px-4 py-2 rounded-lg border border-accent/30">
                    <p className="text-sm font-semibold text-accent">
                      {videoResults.actions?.length || 0} Actions Detected
                    </p>
                  </div>
                </div>

                <div className="p-8 bg-card/30 border-t border-border/20">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {videoResults.actions?.map((action, idx) => (
                      <div
                        key={idx}
                        className="group p-4 rounded-lg border border-border/30 hover:border-accent/50 hover:bg-accent/5 transition-all duration-300"
                      >
                        <div className="flex items-start justify-between mb-3">
                          <h4 className="text-lg font-semibold capitalize">{action.label}</h4>
                          <span className="px-3 py-1 rounded-full bg-gradient-to-r from-accent/20 to-primary/20 text-accent text-xs font-bold">
                            {(action.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="space-y-2">
                          <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-accent to-primary transition-all duration-500"
                              style={{ width: `${action.confidence * 100}%` }}
                            />
                          </div>
                          {action.timestamp && (
                            <p className="text-xs text-muted-foreground">
                              Detected at <span className="font-mono text-accent">{action.timestamp.toFixed(2)}s</span>
                            </p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>

                  {videoResults.summary && (
                    <div className="mt-6 p-6 rounded-xl bg-gradient-to-br from-accent/10 to-primary/10 border border-accent/30">
                      <p className="text-sm font-semibold text-accent mb-2">Summary</p>
                      <p className="text-foreground/80">{videoResults.summary}</p>
                    </div>
                  )}

                  {videoResults.total_duration && (
                    <div className="mt-4 text-sm text-muted-foreground">
                      <span className="font-semibold">Duration:</span> {videoResults.total_duration.toFixed(2)}s
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {searchResults && !searchResults.error && (
          <div className="space-y-12">
            <div className="border-t border-border/20 pt-12">
              <h2 className="text-4xl font-bold mb-8">Search Results</h2>

              {searchResults.matches && searchResults.matches.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {searchResults.matches.map((match, idx) => (
                    <div
                      key={idx}
                      className="group p-6 rounded-xl border border-border/30 hover:border-primary/50 hover:bg-primary/5 transition-all duration-300 bg-card/30"
                    >
                      <div className="flex items-start justify-between mb-4">
                        <h4 className="text-xl font-semibold">{match.action}</h4>
                        <span className="px-3 py-1 rounded-full bg-gradient-to-r from-primary/20 to-accent/20 text-primary text-sm font-bold">
                          {(match.similarity * 100).toFixed(0)}%
                        </span>
                      </div>
                      <p className="text-muted-foreground mb-4">{match.description}</p>
                      <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-500"
                          style={{ width: `${match.similarity * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="p-12 text-center rounded-xl border border-border/20 bg-card/30">
                  <p className="text-muted-foreground">No matching actions found for "{searchQuery}"</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </main>
  )
}
