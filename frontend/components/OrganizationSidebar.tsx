"use client"

import { motion, AnimatePresence } from "framer-motion"
import { X, LogOut, Building2, Users, Calendar, CreditCard } from "lucide-react"
import { useRouter } from "next/navigation"
import { AnimatedButton } from "@/components/ui/animated-button"

interface OrganizationSidebarProps {
  isOpen: boolean
  onClose: () => void
  organization: {
    name: string
    industry: string
    country: string
    logo?: string
    subscription: {
      plan: string
      status: string
      endDate: string
    }
  }
}

export function OrganizationSidebar({ isOpen, onClose, organization }: OrganizationSidebarProps) {
  const router = useRouter()

  const handleLogout = () => {
    localStorage.removeItem("token")
    localStorage.removeItem("user")
    localStorage.removeItem("organization")
    router.push("/auth")
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/50 z-40"
          />

          {/* Sidebar */}
          <motion.div
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 20 }}
            className="fixed right-0 top-0 h-full w-80 bg-background border-l shadow-lg z-50"
          >
            <div className="flex flex-col h-full">
              {/* Header */}
              <div className="flex items-center justify-between p-4 border-b">
                <h2 className="text-lg font-semibold">Organization Details</h2>
                <button
                  onClick={onClose}
                  className="p-2 hover:bg-muted rounded-full"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              {/* Content */}
              <div className="flex-1 overflow-y-auto p-4 space-y-6">
                {/* Logo */}
                <div className="flex justify-center">
                  {organization.logo ? (
                    <img
                      src={organization.logo}
                      alt={organization.name}
                      className="w-24 h-24 rounded-full object-cover"
                    />
                  ) : (
                    <div className="w-24 h-24 rounded-full bg-muted flex items-center justify-center">
                      <Building2 className="h-12 w-12 text-muted-foreground" />
                    </div>
                  )}
                </div>

                {/* Organization Info */}
                <div className="space-y-4">
                  <div className="space-y-2">
                    <h3 className="font-medium">Organization Name</h3>
                    <p className="text-muted-foreground">{organization.name}</p>
                  </div>

                  <div className="space-y-2">
                    <h3 className="font-medium">Industry</h3>
                    <p className="text-muted-foreground capitalize">{organization.industry}</p>
                  </div>

                  <div className="space-y-2">
                    <h3 className="font-medium">Country</h3>
                    <p className="text-muted-foreground">{organization.country}</p>
                  </div>

                  <div className="space-y-2">
                    <h3 className="font-medium">Subscription</h3>
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        organization.subscription.status === "active"
                          ? "bg-green-100 text-green-800"
                          : "bg-red-100 text-red-800"
                      }`}>
                        {organization.subscription.plan.charAt(0).toUpperCase() + organization.subscription.plan.slice(1)}
                      </span>
                      <span className="text-muted-foreground text-sm">
                        {organization.subscription.status === "active" ? "Active" : "Inactive"}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Footer */}
              <div className="p-4 border-t">
                <AnimatedButton
                  onClick={handleLogout}
                  className="w-full bg-red-500 hover:bg-red-600 text-white"
                >
                  <LogOut className="h-4 w-4 mr-2" />
                  Logout
                </AnimatedButton>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
} 