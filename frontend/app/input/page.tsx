"use client"

import ProtectedRoute from "@/components/ProtectedRoute"

export default function InputPage() {
  return (
    <ProtectedRoute>
      <div className="container mx-auto p-6">
        <h1 className="text-2xl font-bold mb-4">Input</h1>
        {/* Add your input page content here */}
      </div>
    </ProtectedRoute>
  );
}

