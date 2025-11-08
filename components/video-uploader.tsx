"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"

interface VideoUploaderProps {
  onAnalyze: (results: any) => void
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
}

export default function VideoUploader({ onAnalyze, isLoading, setIsLoading }: VideoUploaderProps) {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string>("")
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)

      // Create preview URL
      const previewUrl = URL.createObjectURL(selectedFile)
      setPreview(previewUrl)
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setIsLoading(true)
    try {
      const formData = new FormData()
      formData.append("video", file)

      const response = await fetch("/api/analyze-video", {
        method: "POST",
        body: formData,
      })

      const data = await response.json()
      onAnalyze(data)
    } catch (error) {
      console.error("Upload error:", error)
      onAnalyze({
        error: "Failed to analyze video",
        success: false,
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    const droppedFile = e.dataTransfer.files?.[0]
    if (droppedFile && droppedFile.type.startsWith("video/")) {
      setFile(droppedFile)
      const previewUrl = URL.createObjectURL(droppedFile)
      setPreview(previewUrl)
    }
  }

  return (
    <div className="space-y-6">
      <Card className="p-8 border-2 border-dashed border-border hover:border-primary/50 transition-colors">
        <div onDragOver={handleDragOver} onDrop={handleDrop} className="text-center">
          {preview ? (
            <div className="space-y-4">
              <video src={preview} className="w-full h-auto rounded-lg max-h-96 object-cover video-preview" controls />
              <p className="text-sm text-muted-foreground">{file?.name}</p>
            </div>
          ) : (
            <div className="py-8 space-y-4">
              <svg
                className="w-12 h-12 mx-auto text-muted-foreground"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3v-7"
                />
              </svg>
              <div>
                <p className="text-lg font-semibold">Drop your video here</p>
                <p className="text-sm text-muted-foreground">or click to browse (MP4, WebM, etc.)</p>
              </div>
            </div>
          )}
          <input ref={fileInputRef} type="file" accept="video/*" onChange={handleFileChange} className="hidden" />
        </div>
      </Card>

      <Button onClick={() => fileInputRef.current?.click()} variant="outline" className="w-full">
        {file ? "Change Video" : "Select Video"}
      </Button>

      <Button
        onClick={handleUpload}
        disabled={!file || isLoading}
        className="w-full bg-primary hover:bg-primary/90"
        size="lg"
      >
        {isLoading ? (
          <>
            <span className="animate-spin mr-2">⚙️</span>
            Analyzing...
          </>
        ) : (
          "Analyze Video"
        )}
      </Button>
    </div>
  )
}
