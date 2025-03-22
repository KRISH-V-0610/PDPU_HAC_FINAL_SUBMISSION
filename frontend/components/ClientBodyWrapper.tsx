"use client"

import { useEffect } from "react"

export function ClientBodyWrapper({
  children,
}: {
  children: React.ReactNode
}) {
  useEffect(() => {
    // Remove any VS Code related classes that might be added by extensions
    const body = document.body;
    if (body.classList.contains('vsc-initialized')) {
      body.classList.remove('vsc-initialized');
    }
  }, []);

  return <>{children}</>;
} 