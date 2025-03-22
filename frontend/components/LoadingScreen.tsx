"use client"

import { FileText } from "lucide-react"

export function LoadingScreen() {
  return (
    <div className="fixed inset-0 bg-background flex flex-col items-center justify-center z-50">
      <div className="flex flex-col items-center gap-4">
        <div className="flex items-center gap-2">
          <FileText className="h-8 w-8 text-primary animate-pulse" />
          <span className="text-2xl font-bold">ComplianceTrack</span>
        </div>
        <div className="flex flex-col items-center gap-2">
          <div className="h-6 w-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    </div>
  )
} 