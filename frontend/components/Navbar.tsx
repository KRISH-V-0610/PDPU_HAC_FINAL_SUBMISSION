"use client"

import { useState, useEffect, useRef } from "react"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { FileText, Home, Upload, Bell, LogOut, Building2, BarChart3, ArrowUpRight, Camera, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { ThemeToggle } from "@/components/ThemeToggle"
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { useToast } from "@/components/ui/use-toast"
import { Progress } from "@/components/ui/progress"

const MAX_FILE_SIZE = 10 * 1024 * 1024 // 10MB in bytes

export default function Navbar() {
  const pathname = usePathname()
  const router = useRouter()
  const [organization, setOrganization] = useState<any>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { toast } = useToast()

  useEffect(() => {
    // Get organization data from localStorage when component mounts
    const orgData = localStorage.getItem("organization")
    if (orgData) {
      setOrganization(JSON.parse(orgData))
    }
  }, [])

  // Reset progress when upload completes
  useEffect(() => {
    if (!uploading) {
      const timer = setTimeout(() => {
        setUploadProgress(0)
      }, 2000)
      return () => clearTimeout(timer)
    }
  }, [uploading])

  const handleLogout = () => {
    localStorage.removeItem("token")
    localStorage.removeItem("user")
    localStorage.removeItem("organization")
    router.push("/auth")
  }

  const handlePhotoClick = () => {
    fileInputRef.current?.click()
  }

  const simulateProgress = () => {
    setUploadProgress(0)
    const interval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 90) {
          clearInterval(interval)
          return 90
        }
        return prev + 10
      })
    }, 200)
    return interval
  }

  const handlePhotoChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return

    const file = e.target.files[0]

    // Check file size
    if (file.size > MAX_FILE_SIZE) {
      toast({
        title: "Error",
        description: "File size must be less than 10MB",
        variant: "destructive",
      })
      return
    }

    setUploading(true)
    const progressInterval = simulateProgress()

    toast({
      title: "Uploading...",
      description: "Please wait while we upload your photo",
    })

    try {
      // First, upload to Cloudinary
      const formData = new FormData()
      formData.append("file", file)
      formData.append("upload_preset", "organization_photos") // Make sure this matches your Cloudinary upload preset

      const cloudinaryResponse = await fetch(
        `https://api.cloudinary.com/v1_1/${process.env.NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME}/image/upload`,
        {
          method: "POST",
          body: formData,
        }
      )

      if (!cloudinaryResponse.ok) {
        throw new Error("Failed to upload to Cloudinary")
      }

      const cloudinaryData = await cloudinaryResponse.json()
      setUploadProgress(95)

      // Now update the organization profile with the Cloudinary URL
      const updateResponse = await fetch("/api/organization/profile", {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${localStorage.getItem("token")}`,
        },
        body: JSON.stringify({
          profileImageUrl: cloudinaryData.secure_url,
        }),
      })

      if (!updateResponse.ok) {
        throw new Error("Failed to update organization profile")
      }

      setUploadProgress(100)
      const data = await updateResponse.json()

      // Update organization in localStorage and state
      const updatedOrg = { ...organization, profileImageUrl: cloudinaryData.secure_url }
      localStorage.setItem("organization", JSON.stringify(updatedOrg))
      setOrganization(updatedOrg)

      toast({
        title: "Success! 🎉",
        description: "Profile photo updated successfully",
        variant: "default",
      })
    } catch (error) {
      console.error("Upload error:", error)
      toast({
        title: "Error! ❌",
        description: error instanceof Error ? error.message : "Failed to upload profile photo",
        variant: "destructive",
      })
    } finally {
      clearInterval(progressInterval)
      setUploading(false)
      if (fileInputRef.current) {
        fileInputRef.current.value = ""
      }
    }
  }

  return (
    <header className="sticky top-0 z-10 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between">
        <div className="flex items-center gap-2">
          <FileText className="h-6 w-6 text-primary" />
          <span className="text-xl font-bold">ComplySmart</span>
        </div>
        <nav className="flex items-center gap-6">
          <Link href="/" className="flex items-center gap-1 font-medium">
            <Home className="h-4 w-4" />
            Home
          </Link>
          <Link href="/dashboard" className="flex items-center gap-1 font-medium">
            <BarChart3 className="h-4 w-4" />
            Dashboard
          </Link>
          <Link href="/notifications" className="flex items-center gap-1 font-medium">
            <Bell className="h-4 w-4" />
            Notifications
          </Link>
          <Button asChild>
            <Link href="/upload" className="text-primary-foreground">
              <ArrowUpRight className="mr-2 h-4 w-4" />
              Upload Report
            </Link>
          </Button>
          <Sheet>
            <SheetTrigger asChild>
              <Button variant="outline" size="sm" className="flex items-center gap-2">
                <Building2 className="h-4 w-4" />
                Organization
              </Button>
            </SheetTrigger>
            <SheetContent>
              <SheetHeader>
                <SheetTitle>Organization Profile</SheetTitle>
              </SheetHeader>
              <div className="mt-6 space-y-6">
                <div className="flex flex-col items-center space-y-4">
                  <div className="relative">
                    <Avatar className="h-24 w-24 ring-2 ring-primary ring-offset-2">
                      <AvatarImage 
                        src={organization?.profileImageUrl} 
                        className="object-cover"
                      />
                      <AvatarFallback>
                        <Building2 className="h-12 w-12" />
                      </AvatarFallback>
                    </Avatar>
                    <Button
                      size="icon"
                      variant="secondary"
                      className="absolute bottom-0 right-0 rounded-full h-8 w-8 shadow-lg"
                      onClick={handlePhotoClick}
                      disabled={uploading}
                    >
                      {uploading ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Camera className="h-4 w-4" />
                      )}
                      <span className="sr-only">
                        {uploading ? "Uploading..." : "Upload photo"}
                      </span>
                    </Button>
                    <input
                      type="file"
                      ref={fileInputRef}
                      className="hidden"
                      accept="image/*"
                      onChange={handlePhotoChange}
                    />
                  </div>
                  {uploading && (
                    <div className="w-full space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Uploading...</span>
                        <span className="text-muted-foreground">{uploadProgress}%</span>
                      </div>
                      <Progress value={uploadProgress} className="h-2" />
                    </div>
                  )}
                  <div className="text-center">
                    <h3 className="text-lg font-semibold">{organization?.name || "Organization Name"}</h3>
                    <p className="text-sm text-muted-foreground">{organization?.industry || "Industry"}</p>
                  </div>
                </div>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span className="text-sm font-medium">Country</span>
                    <span className="text-sm text-muted-foreground">{organization?.country || "Country"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm font-medium">Subscription Plan</span>
                    <span className="text-sm text-muted-foreground capitalize">
                      {organization?.subscription?.plan || "Free"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm font-medium">Status</span>
                    <span className={`text-sm capitalize ${
                      organization?.subscription?.status === "active" ? "text-green-500" : "text-red-500"
                    }`}>
                      {organization?.subscription?.status || "inactive"}
                    </span>
                  </div>
                </div>
                <Button
                  variant="destructive"
                  className="w-full"
                  onClick={handleLogout}
                >
                  <LogOut className="mr-2 h-4 w-4" />
                  Logout
                </Button>
              </div>
            </SheetContent>
          </Sheet>
          <ThemeToggle />
        </nav>
      </div>
    </header>
  )
} 