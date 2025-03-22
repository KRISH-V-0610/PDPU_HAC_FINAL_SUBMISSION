"use client"

import { useEffect, useState } from "react"
import { useRouter, usePathname } from "next/navigation"

interface AuthGuardProps {
  children: React.ReactNode
}

const PUBLIC_PATHS = ["/auth"]

export function AuthGuard({ children }: AuthGuardProps) {
  const router = useRouter()
  const pathname = usePathname()
  const [isAuthenticated, setIsAuthenticated] = useState(false)

  useEffect(() => {
    // Check if we're on a public path
    if (PUBLIC_PATHS.includes(pathname)) {
      setIsAuthenticated(true)
      return
    }

    // Check for token
    const token = localStorage.getItem("token")
    const organization = localStorage.getItem("organization")

    if (!token || !organization) {
      router.push("/auth")
      return
    }

    // Verify token expiration
    try {
      const orgData = JSON.parse(organization)
      setIsAuthenticated(true)
    } catch (error) {
      console.error("Error parsing organization data:", error)
      localStorage.removeItem("token")
      localStorage.removeItem("organization")
      router.push("/auth")
    }
  }, [pathname, router])

  // Show nothing while checking authentication
  if (!isAuthenticated) {
    return null
  }

  // If authenticated or on public path, show the page
  return <>{children}</>
} 