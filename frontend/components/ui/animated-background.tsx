import { motion, useAnimation } from "framer-motion";
import { useEffect, useState } from "react";

const AnimatedBackground = () => {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [clickPoints, setClickPoints] = useState<{ x: number; y: number }[]>(
    []
  );
  const [windowSize, setWindowSize] = useState({ width: 0, height: 0 });
  const controls = useAnimation();

  useEffect(() => {
    if (typeof window !== "undefined") {
      setWindowSize({ width: window.innerWidth, height: window.innerHeight });

      const handleResize = () => {
        setWindowSize({ width: window.innerWidth, height: window.innerHeight });
      };

      window.addEventListener("resize", handleResize);
      return () => window.removeEventListener("resize", handleResize);
    }
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };

    const handleClick = (e: MouseEvent) => {
      if ((e.target as HTMLElement).classList.contains("animated-background")) {
        setClickPoints((prev) => [
          ...prev.slice(-4),
          { x: e.clientX, y: e.clientY },
        ]);
      }
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("click", handleClick);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("click", handleClick);
    };
  }, []);

  return (
    <div className="fixed inset-0 -z-10 overflow-hidden bg-background animated-background">
      {/* Click effect circles */}
      {clickPoints.map((point, i) => (
        <motion.div
          key={`${point.x}-${point.y}-${i}`}
          className="absolute h-[500px] w-[500px] rounded-full bg-primary/5"
          initial={{ scale: 0, x: point.x - 250, y: point.y - 250 }}
          animate={{ scale: 1, opacity: [0, 0.5, 0] }}
          transition={{ duration: 1.5, ease: "easeOut" }}
        />
      ))}

      {[...Array(8)].map((_, i) => {
        // Generate random initial positions for each box
        const initialX = Math.random() * windowSize.width;
        const initialY = Math.random() * windowSize.height;

        return (
          <motion.div
            key={i}
            className="absolute h-32 w-32 rounded-lg bg-primary/5"
            initial={{
              x: initialX, // Random initial x position
              y: initialY, // Random initial y position
              scale: 0,
              rotate: 0,
            }}
            animate={{
              x: Math.random() * windowSize.width, // Move to a new random x position
              y: Math.random() * windowSize.height, // Move to a new random y position
              scale: [0, 1, 0],
              rotate: 360,
            }}
            transition={{
              duration: 8 + Math.random() * 6,
              repeat: Infinity,
              ease: "easeInOut",
              delay: i * 0.5, // Stagger the emergence of boxes
            }}
          />
        );
      })}
      {/* Mouse gradient effect */}
      <motion.div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: `radial-gradient(600px circle at ${mousePosition.x}px ${mousePosition.y}px, rgba(var(--primary), 0.1) 0%, transparent 60%)`,
        }}
      />

      {/* Animated paths */}
      <svg className="absolute inset-0 w-full h-full opacity-30">
        <motion.path
          d={`M 0,${windowSize.height / 2} Q ${windowSize.width / 4},${
            windowSize.height / 2 - 100
          } ${windowSize.width / 2},${windowSize.height / 2} T ${
            windowSize.width
          },${windowSize.height / 2}`}
          stroke="var(--primary)"
          strokeWidth="1"
          fill="none"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.path
          d={`M 0,${windowSize.height / 2 + 50} Q ${windowSize.width / 4},${
            windowSize.height / 2 - 50
          } ${windowSize.width / 2},${windowSize.height / 2 + 50} T ${
            windowSize.width
          },${windowSize.height / 2 + 50}`}
          stroke="var(--primary)"
          strokeWidth="1"
          fill="none"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 1,
          }}
        />
      </svg>
    </div>
  );
};

export default AnimatedBackground;
