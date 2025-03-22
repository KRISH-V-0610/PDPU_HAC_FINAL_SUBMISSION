"use client"

import { createContext, useContext, useState, useEffect } from "react"
import { usePathname, useSearchParams } from "next/navigation"
import { LoadingScreen } from "@/components/LoadingScreen"

interface NavigationContextType {
  isLoading: boolean
  startLoading: () => void
  stopLoading: () => void
}

const NavigationContext = createContext<NavigationContextType>({
  isLoading: false,
  startLoading: () => {},
  stopLoading: () => {},
})

export function useNavigation() {
  return useContext(NavigationContext)
}

export function NavigationProvider({ children }: { children: React.ReactNode }) {
  const [isLoading, setIsLoading] = useState(false)
  const pathname = usePathname()
  const searchParams = useSearchParams()

  useEffect(() => {
    // Stop loading when route changes
    setIsLoading(false)
  }, [pathname, searchParams])

  const startLoading = () => setIsLoading(true)
  const stopLoading = () => setIsLoading(false)

  return (
    <NavigationContext.Provider value={{ isLoading, startLoading, stopLoading }}>
      {isLoading && <LoadingScreen />}
      {children}
    </NavigationContext.Provider>
  )
} 