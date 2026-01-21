import { useState } from "react"
import { ECGScrollDemo } from "@/components/ECGScrollDemo"
import { FeatureGrid } from "@/components/FeatureGrid"
import { ECGUpload } from "@/components/ECGUpload"
import { ResultsDisplay } from "@/components/ResultsDisplay"
import { Activity } from "lucide-react"

function App() {
  const [prediction, setPrediction] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const handleFileSelect = async (file: File) => {
    setLoading(true)
    const formData = new FormData()
    formData.append("file", file)

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      })
      const data = await response.json()
      setPrediction(data)
    } catch (error) {
      console.error("Error:", error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen relative">
      {/* Radial gradient background */}
      <div 
        className="fixed inset-0 z-0"
        style={{
          background: "radial-gradient(125% 125% at 50% 10%, #fff 40%, #FFB5B5 100%)",
        }}
      />
      
      {/* Content wrapper with higher z-index */}
      <div className="relative z-10">
        <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border-b">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-center gap-2">
            <Activity className="h-6 w-6 text-red-500" />
            <span className="text-xl font-bold">CardioAI</span>
          </div>
        </div>
      </nav>

        <ECGScrollDemo />

        <div id="features"><FeatureGrid /></div>
        <div id="upload"><ECGUpload onFileSelect={handleFileSelect} loading={loading} /></div>
        {prediction && <ResultsDisplay prediction={prediction} />}

        <footer className="py-12 bg-white/50 dark:bg-gray-900/50 backdrop-blur-md">
        <div className="container mx-auto px-4 text-center">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Activity className="h-6 w-6 text-red-500" />
            <span className="text-xl font-bold">CardioAI</span>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            For research and educational purposes only. Made in collaboration with the open-source community.
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
            Technical University of Cluj-Napoca • Stefan Brad • 2026
          </p>
        </div>
      </footer>
      </div>
    </div>
  )
}

export default App