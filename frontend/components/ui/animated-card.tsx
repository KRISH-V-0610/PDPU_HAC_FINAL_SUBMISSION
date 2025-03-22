"use client"

import * as React from "react"
import { motion, HTMLMotionProps } from "framer-motion"
import { cn } from "@/lib/utils"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./card"

interface AnimatedCardProps extends HTMLMotionProps<"div"> {
  children: React.ReactNode
  className?: string
  title?: string
  description?: string
  as?: React.ElementType
}

export function AnimatedCard({ 
  children, 
  className,
  title,
  description,
  as: Component = motion.div,
  ...props 
}: AnimatedCardProps) {
  return (
    <Component
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      whileHover={{ 
        y: -5,
        transition: { type: "spring", stiffness: 300, damping: 15 }
      }}
      className={cn(
        "relative overflow-hidden",
        className
      )}
    >
      <div className="absolute inset-0 bg-gradient-to-r from-primary/0 via-primary/5 to-primary/0 opacity-0 transition-opacity duration-500 hover:opacity-100" />
      {(title || description) && (
        <CardHeader>
          {title && <CardTitle>{title}</CardTitle>}
          {description && <CardDescription>{description}</CardDescription>}
        </CardHeader>
      )}
      <CardContent>{children}</CardContent>
    </Component>
  )
} 