"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface ResultsDisplayProps {
  results: any
  type: "video" | "search"
}

export default function ResultsDisplay({ results, type }: ResultsDisplayProps) {
  if (results.error) {
    return (
      <Card className="p-6 bg-destructive/10 border-destructive/20">
        <p className="text-destructive font-semibold">{results.error}</p>
      </Card>
    )
  }

  if (type === "video") {
    return (
      <div className="space-y-4">
        <h2 className="text-xl font-bold">Analysis Results</h2>

        <Card className="p-6 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {results.actions &&
              results.actions.map((action: any, idx: number) => (
                <div key={idx} className="p-4 bg-card border border-border rounded-lg space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold text-balance">{action.label}</h3>
                    <Badge className="action-badge bg-primary/20 text-primary border-0">
                      {(action.confidence * 100).toFixed(1)}%
                    </Badge>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-muted-foreground">Confidence</span>
                      <span className="font-semibold">{(action.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-muted rounded-full overflow-hidden h-2">
                      <div className="confidence-bar" style={{ width: `${action.confidence * 100}%` }} />
                    </div>
                  </div>

                  {action.timestamp && (
                    <p className="text-xs text-muted-foreground">Time: {action.timestamp.toFixed(2)}s</p>
                  )}
                </div>
              ))}
          </div>

          {results.summary && (
            <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <p className="text-sm">
                <span className="font-semibold">Summary:</span> {results.summary}
              </p>
            </div>
          )}

          {results.total_duration && (
            <div className="text-sm text-muted-foreground">
              <span className="font-semibold">Duration:</span> {results.total_duration.toFixed(2)}s
            </div>
          )}
        </Card>
      </div>
    )
  }

  // Search results
  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold">Search Results</h2>

      {results.matches && results.matches.length > 0 ? (
        <div className="space-y-3">
          {results.matches.map((match: any, idx: number) => (
            <Card key={idx} className="p-4 space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold">{match.action}</h3>
                <Badge className="action-badge bg-primary/20 text-primary border-0">
                  {(match.similarity * 100).toFixed(1)}% Match
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground">{match.description}</p>
              <div className="w-full bg-muted rounded-full overflow-hidden h-2">
                <div className="confidence-bar" style={{ width: `${match.similarity * 100}%` }} />
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <Card className="p-6 text-center text-muted-foreground">
          <p>No matching actions found</p>
        </Card>
      )}

      {results.query && (
        <p className="text-xs text-muted-foreground">
          <span className="font-semibold">Query:</span> {results.query}
        </p>
      )}
    </div>
  )
}
