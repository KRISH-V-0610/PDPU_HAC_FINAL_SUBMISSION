"use client"

import type React from "react"
import { useState, useRef } from "react"
import Link from "next/link"
import { ArrowLeft, Upload, Loader2, FileText } from "lucide-react" // Import FileText
import { Toaster, toast } from 'react-hot-toast'

import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { ThemeToggle } from "@/components/ThemeToggle"
import { FileIcon } from "@/components/ui/file-icon"
import ProtectedRoute from "@/components/ProtectedRoute"
import { useAuth } from "@/context/AuthContext"

interface UploadedFile {
  name: string;
  size: number;
  type: string;
  progress: number;
  status: 'uploading' | 'success' | 'error';
  error?: string;
}

export default function UploadPage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const { token } = useAuth();

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files).map(file => ({
        name: file.name,
        size: file.size,
        type: file.type,
        progress: 0,
        status: 'uploading' as const
      }));
      setFiles(prev => [...prev, ...newFiles]);
      uploadAllFiles(Array.from(e.target.files));
    }
  };

  const uploadFileAndFetchAuditData = async (formData: FormData) => {
    try {
      // First fetch: Upload the file
      const response = await fetch('http://localhost:5000/api/files/upload', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      // Check if the file upload was successful
      if (!response.ok) {
        throw new Error(`File upload failed: ${response.statusText}`);
      }

      // Parse the response from the file upload
      const data = await response.json();
      console.log('File upload response:', data);

      // Second fetch: Fetch audit data only after the file is uploaded
      const getAuditData = await fetch('https://fingenius-ai-fastapi-0rad.onrender.com/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // Add a body if needed (e.g., JSON payload)
        // body: JSON.stringify({ someKey: 'someValue' }),
      });

      // Check if the audit data fetch was successful
      if (!getAuditData.ok) {
        throw new Error(`Audit data fetch failed: ${getAuditData.statusText}`);
      }

      // Parse the response from the audit data fetch
      const auditData = await getAuditData.json();
      console.log('Audit data response:', auditData);

      return data; // Return the response data for further use
    } catch (error) {
      console.error('Error:', error);
      throw error; // Re-throw the error for handling in the calling function
    }
  };





  const uploadFile = async (file: File, index: number) => {
    try {
      if (!token) {
        throw new Error('Authentication token is missing');
      }

      const formData = new FormData();
      formData.append('file', file);

      // Upload file and fetch audit data
      const data = await uploadFileAndFetchAuditData(formData);

      // Update file status to success
      setFiles(prev => prev.map((f, i) => 
        i === index ? { ...f, status: 'success', progress: 100 } : f
      ));

      toast.success(`Successfully uploaded ${file.name}`);
      return data;
    } catch (error) {
      console.error('Upload error:', error);
      // Update file status to error
      setFiles(prev => prev.map((f, i) => 
        i === index ? { 
          ...f, 
          status: 'error',
          error: error instanceof Error ? error.message : 'Upload failed'
        } : f
      ));

      toast.error(`Failed to upload ${file.name}: ${error instanceof Error ? error.message : 'Unknown error'}`);
      throw error;
    }
  };

  const uploadAllFiles = async (filesToUpload: File[]) => {
    setIsUploading(true);
    try {
      await Promise.all(
        filesToUpload.map((file, index) => uploadFile(file, index))
      );
    } catch (error) {
      console.error('Error uploading files:', error);
    } finally {
      setIsUploading(false);
    }
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  return (
    <ProtectedRoute>
      
      <div className="flex min-h-screen flex-col">
        <Toaster position="top-right" />
        <header className="sticky top-0 z-10 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="container flex h-16 items-center justify-between">
            <div className="flex items-center gap-2">
              <FileText className="h-6 w-6 text-primary" /> {/* Use FileText here */}
              <span className="text-xl font-bold">ComplySmart</span>
            </div>
            <ThemeToggle />
          </div>
        </header>

        <main className="flex-1">
          <div className="container py-12">
            <div className="mx-auto max-w-2xl space-y-8">
              <div className="flex items-center gap-2">
                <Button variant="ghost" size="icon" asChild>
                  <Link href="/">
                    <ArrowLeft className="h-4 w-4" />
                    <span className="sr-only">Back to dashboard</span>
                  </Link>
                </Button>
                <h1 className="text-3xl font-bold tracking-tighter">Upload Files</h1>
              </div>

              <Card>
                <CardContent className="pt-6">
                  <div className="space-y-6">
                    <div className="flex flex-col items-center justify-center gap-4 rounded-lg border border-dashed p-8 text-center">
                      <Upload className="h-8 w-8 text-muted-foreground" />
                      <div className="space-y-1">
                        <h3 className="text-lg font-medium">Upload your files</h3>
                        <p className="text-sm text-muted-foreground">
                          Drag and drop your files here, or click to select files
                        </p>
                      </div>
                      <input
                        ref={fileInputRef}
                        type="file"
                        className="hidden"
                        multiple
                        onChange={handleFileSelect}
                        accept=".pdf,.doc,.docx,.xls,.xlsx"
                      />
                      <Button
                        variant="outline"
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isUploading}
                      >
                        Select Files
                      </Button>
                    </div>

                    {files.length > 0 && (
                      <div className="space-y-4">
                        <h3 className="text-lg font-medium">Selected Files</h3>
                        <div className="space-y-2">
                          {files.map((file, index) => (
                            <div
                              key={index}
                              className="flex items-center justify-between rounded-lg border p-4"
                            >
                              <div className="flex items-center gap-3">
                                <FileIcon type={file.type} />
                                <div className="space-y-1">
                                  <p className="text-sm font-medium">{file.name}</p>
                                  <p className="text-xs text-muted-foreground">
                                    {(file.size / 1024 / 1024).toFixed(2)} MB
                                  </p>
                                </div>
                              </div>
                              <div className="flex items-center gap-2">
                                {file.status === 'uploading' && (
                                  <Loader2 className="h-4 w-4 animate-spin" />
                                )}
                               {file.status === 'success' && (
  <>
    <span className="text-green-500">✓</span>
    {(() => {
      window.location.href = "https://complysmart-dashboard.streamlit.app/";
      return null; // Prevent rendering anything
    })()}
  </>
)}

                                {file.status === 'error' && (
                                  <span className="text-red-500">✕</span>
                                )}
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => removeFile(index)}
                                  disabled={file.status === 'uploading'}
                                >
                                  <span className="sr-only">Remove file</span>
                                  <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    width="24"
                                    height="24"
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    stroke="currentColor"
                                    strokeWidth="2"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    className="h-4 w-4"
                                  >
                                    <path d="M18 6 6 18" />
                                    <path d="m6 6 12 12" />
                                  </svg>
                                </Button>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </main>

        <footer className="border-t py-6">
          <div className="container flex flex-col items-center justify-between gap-4 md:flex-row">
            <p className="text-center text-sm text-muted-foreground">
              &copy; {new Date().getFullYear()} ComplianceTrack. All rights reserved.
            </p>
            <p className="text-center text-sm text-muted-foreground">
              Designed for efficient compliance management
            </p>
          </div>
        </footer>
      </div>
    </ProtectedRoute>
  );
}