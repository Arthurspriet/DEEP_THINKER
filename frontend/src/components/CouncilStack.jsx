export default function CouncilStack({ models, isActive }) {
  return (
    <div className="flex flex-wrap justify-center gap-1.5 max-w-[160px]">
      {models.map((model, index) => (
        <div
          key={index}
          className={`
            px-2 py-1 rounded-md text-xs font-mono
            ${isActive
              ? 'bg-dt-accent/20 text-dt-accent border border-dt-accent/40 animate-pulse'
              : 'bg-dt-surface-light text-dt-text-dim border border-dt-border'
            }
            transition-all duration-300
          `}
          style={{ animationDelay: `${index * 100}ms` }}
        >
          {formatModelName(model)}
        </div>
      ))}
    </div>
  )
}

function formatModelName(name) {
  // Shorten model names for display
  if (name.length <= 10) return name
  
  // Remove version numbers and size indicators for compactness
  const parts = name.split(':')
  const base = parts[0]
  
  if (base.length <= 8) return name
  return base.slice(0, 8) + '...'
}

