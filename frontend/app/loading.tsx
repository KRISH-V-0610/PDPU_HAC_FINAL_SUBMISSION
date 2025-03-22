"use client"

import AnimatedBackground from "@/components/ui/animated-background"
import { Loader2 } from "lucide-react"

export default function Loading() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center">
      <AnimatedBackground />
      <div className="relative z-10 flex flex-col items-center gap-4">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <p className="text-lg font-medium text-muted-foreground">Loading...</p>
      </div>
    </div>
  )
} 