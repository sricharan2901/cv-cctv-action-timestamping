"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"

interface SearchBarProps {
  onSearch: (results: any) => void
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
}

export default function SearchBar({ onSearch, isLoading, setIsLoading }: SearchBarProps) {
  const [query, setQuery] = useState("")

  const handleSearch = async () => {
    if (!query.trim()) return

    setIsLoading(true)
    try {
      const response = await fetch("/api/search-action", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: query.trim() }),
      })

      const data = await response.json()
      onSearch(data)
    } catch (error) {
      console.error("Search error:", error)
      onSearch({
        error: "Failed to search",
        success: false,
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch()
    }
  }

  return (
    <Card className="p-6 space-y-4">
      <div>
        <label className="block text-sm font-medium mb-2">Search Action</label>
        <div className="flex gap-2">
          <Input
            placeholder="Enter action to search (e.g., running, falling, fighting)"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            className="flex-1"
          />
          <Button
            onClick={handleSearch}
            disabled={!query.trim() || isLoading}
            className="bg-primary hover:bg-primary/90 min-w-24"
          >
            {isLoading ? (
              <>
                <span className="animate-spin mr-2">⚙️</span>
                <span className="hidden sm:inline">Searching</span>
              </>
            ) : (
              "Search"
            )}
          </Button>
        </div>
      </div>
    </Card>
  )
}
