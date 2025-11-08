export async function POST(request) {
  try {
    const body = await request.json()
    const { query } = body

    if (!query || !query.trim()) {
      return Response.json({ error: "No search query provided", success: false }, { status: 400 })
    }

    // REPLACE THIS SECTION with your ML model API call
    // Example: const response = await fetch(process.env.SEARCH_MODEL_URL, { ... })

    // Mock response for testing
    const mockResults = {
      success: true,
      query: query.trim(),
      matches: [
        {
          action: "Running",
          similarity: 0.95,
          description: "High-speed movement detected. Person moving rapidly across the frame.",
        },
        {
          action: "Sprinting",
          similarity: 0.89,
          description: "Fast-paced movement. Similar to running but with higher acceleration.",
        },
        {
          action: "Jogging",
          similarity: 0.82,
          description: "Moderate-speed movement. Slower than sprinting but faster than walking.",
        },
      ],
    }

    return Response.json(mockResults)
  } catch (error) {
    console.error("Error searching actions:", error)
    return Response.json({ error: "Failed to search actions", success: false }, { status: 500 })
  }
}
