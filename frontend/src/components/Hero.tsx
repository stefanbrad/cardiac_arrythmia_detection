import { Button } from "@/components/ui/button"
import { Upload, Activity } from "lucide-react"

interface HeroProps {
  onUploadClick: () => void
}

export function Hero({ onUploadClick }: HeroProps) {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* NEW: Radial gradient background */}
      <div 
        className="absolute inset-0 z-0"
        style={{
          background: "radial-gradient(125% 125% at 50% 10%, #fff 40%, #6366f1 100%)",
        }}
      />
      
      <div className="container mx-auto px-4 relative z-10">
        <div className="max-w-4xl mx-auto text-center space-y-8">
          <div className="flex justify-center">
            <div className="relative">
              <div className="absolute inset-0 bg-blue-500 blur-3xl opacity-30 animate-pulse" />
              <Activity className="h-20 w-20 text-blue-600 dark:text-blue-400 relative" strokeWidth={1.5} />
            </div>
          </div>

          <h1 className="text-5xl md:text-7xl font-bold tracking-tight">
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">
              Cardiac Arrhythmia
            </span>
            <br />
            <span className="text-gray-900 dark:text-white">
              Detection
            </span>
          </h1>

          <p className="text-xl md:text-2xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            Advanced AI-powered ECG analysis for accurate arrhythmia detection
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center pt-4">
            <Button 
              size="lg" 
              className="text-lg px-8 py-6 gap-2"
              onClick={onUploadClick}
            >
              <Upload className="h-5 w-5" />
              Upload ECG Data
            </Button>
            <Button 
              size="lg" 
              variant="outline"
              className="text-lg px-8 py-6"
            >
              Learn More
            </Button>
          </div>

          <div className="flex flex-wrap gap-4 justify-center pt-8 text-sm text-gray-600 dark:text-gray-400">
            <div className="flex items-center gap-2 bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm px-4 py-2 rounded-full border border-gray-200 dark:border-gray-700">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <span>Medical-grade Accuracy</span>
            </div>
            <div className="flex items-center gap-2 bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm px-4 py-2 rounded-full border border-gray-200 dark:border-gray-700">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
              <span>Real-time Analysis</span>
            </div>
            <div className="flex items-center gap-2 bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm px-4 py-2 rounded-full border border-gray-200 dark:border-gray-700">
              <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" />
              <span>Multiple Arrhythmia Types</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}