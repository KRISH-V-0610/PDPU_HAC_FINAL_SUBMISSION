"use client"

import * as React from "react"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"
import { Button } from "./button"
import { Slot } from "@radix-ui/react-slot"
import Link from "next/link"
import { ArrowRight } from "lucide-react"

interface AnimatedButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode
  className?: string
  variant?: "default" | "destructive" | "outline" | "secondary" | "ghost" | "link"
  size?: "default" | "sm" | "lg" | "icon"
  asChild?: boolean
}

export function AnimatedButton({ 
  children, 
  className,
  variant = "default",
  size = "default",
  asChild = false,
  ...props 
}: AnimatedButtonProps) {
  const Comp = asChild ? Slot : motion(Button)
  
  return (
    <motion.div
      whileHover={{ 
        scale: 1.02,
        transition: { 
          type: "spring", 
          stiffness: 400, 
          damping: 10,
          mass: 0.5
        }
      }}
      whileTap={{ 
        scale: 0.98,
        transition: { 
          type: "spring", 
          stiffness: 400, 
          damping: 10,
          mass: 0.5
        }
      }}
    >
      <Comp
        variant={variant}
        size={size}
        className={cn(
          "relative overflow-hidden transition-all duration-300",
          "group",
          "after:absolute after:bottom-0 after:left-0 after:h-0.5 after:w-0 after:bg-primary after:transition-all after:duration-300 hover:after:w-full",
          "shadow-sm hover:shadow-md",
          className
        )}
        {...props}
      >
        {asChild ? (
          <Link href="/upload" className="flex items-center gap-2">
            {children}
            <ArrowRight className="h-4 w-4" />
          </Link>
        ) : children}
      </Comp>
    </motion.div>
  )
} 