import { useState, useCallback } from "react"
import { Upload, FileText, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { GlowingEffect } from "@/components/ui/glowing-effect"

interface ECGUploadProps {
  onFileSelect: (file: File) => void | Promise<void>
  loading?: boolean
}

function getExt(name: string) {
  const parts = name.split(".")
  return parts.length > 1 ? parts.pop()!.toLowerCase() : ""
}

export function ECGUpload({ onFileSelect, loading }: ECGUploadProps) {
  const [dragActive, setDragActive] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const validateAndSetFile = useCallback((file: File) => {
    const ext = getExt(file.name)

    if (ext === "dat") {
      alert(
        "MIT-BIH .dat requires the matching .hea file. Please upload a .zip containing both .dat and .hea (and optionally .atr)."
      )
      setSelectedFile(null)
      return
    }

    const allowed = ["csv", "txt", "json", "zip", "hea", "dat"]
    if (!allowed.includes(ext)) {
      alert("Unsupported file type. Please upload .csv, .txt, .json, or .zip.")
      setSelectedFile(null)
      return
    }

    setSelectedFile(file)
  }, [])

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      setDragActive(false)

      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        validateAndSetFile(e.dataTransfer.files[0])
      }
    },
    [validateAndSetFile]
  )

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      e.preventDefault()
      if (e.target.files && e.target.files[0]) {
        validateAndSetFile(e.target.files[0])
      }
    },
    [validateAndSetFile]
  )

  const removeFile = () => {
    setSelectedFile(null)
  }

  const handleAnalyze = async () => {
    if (!selectedFile) return
    await onFileSelect(selectedFile)
  }

  return (
    <section id="upload-section" className="py-20">
      <div className="container mx-auto px-4 max-w-3xl">
        <Card className="relative overflow-hidden">
          <GlowingEffect
            spread={40}
            glow={true}
            disabled={false}
            proximity={64}
            inactiveZone={0.01}
            borderWidth={2}
          />
          <CardHeader className="text-center">
            <CardTitle className="text-2xl">Upload ECG Data</CardTitle>
            <CardDescription>
              Upload your ECG recording for arrhythmia analysis
            </CardDescription>
          </CardHeader>

          <CardContent>
            <div className="relative">
              <GlowingEffect
                spread={50}
                glow={true}
                disabled={false}
                proximity={80}
                inactiveZone={0.01}
                borderWidth={3}
              />
              <div
                className={`relative border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
                  dragActive
                    ? "border-blue-500 bg-blue-50 dark:bg-blue-950/20"
                    : "border-gray-300 dark:border-gray-700"
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <input
                  type="file"
                  id="file-upload"
                  className="hidden"
                  onChange={handleChange}
                  accept=".csv,.txt,.json,.zip,.dat,.hea"
                />

                {!selectedFile ? (
                  <label htmlFor="file-upload" className="cursor-pointer">
                    <div className="space-y-4">
                      <Upload className="h-12 w-12 mx-auto text-gray-400" />
                      <div>
                        <p className="text-lg font-medium">
                          Drop your ECG file here, or{" "}
                          <span className="text-red-500 hover:text-red-600">
                            browse
                          </span>
                        </p>
                        <p className="text-sm text-gray-500 mt-2">
                          Supports: CSV, TXT, JSON, ZIP (MIT-BIH: ZIP with .dat + .hea)
                        </p>
                      </div>
                    </div>
                  </label>
                ) : (
                  <div className="space-y-4">
                    <FileText className="h-12 w-12 mx-auto text-red-500" />
                    <div className="flex items-center justify-center gap-2">
                      <p className="text-lg font-medium">{selectedFile.name}</p>
                      <button
                        onClick={removeFile}
                        className="text-gray-500 hover:text-red-600"
                        disabled={loading}
                        type="button"
                      >
                        <X className="h-5 w-5" />
                      </button>
                    </div>
                    <p className="text-sm text-gray-500">
                      Size: {(selectedFile.size / 1024).toFixed(2)} KB
                    </p>
                    <Button className="mt-4" onClick={handleAnalyze} disabled={loading}>
                      {loading ? "Analyzing..." : "Analyze ECG"}
                    </Button>
                  </div>
                )}
              </div>
            </div>

            <div className="mt-6 space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <p className="font-medium">Supported formats:</p>
              <ul className="list-disc list-inside space-y-1">
                <li>
                  MIT-BIH: upload a <strong>.zip</strong> containing{" "}
                  <strong>record.dat</strong> + <strong>record.hea</strong> (optional: <strong>record.atr</strong>)
                </li>
                <li>CSV with time and amplitude columns</li>
                <li>Plain text with ECG signal values</li>
                <li>JSON with ECG signal values</li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>
    </section>
  )
}