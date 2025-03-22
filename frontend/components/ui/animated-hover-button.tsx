import { motion } from "framer-motion";
import { Button } from "./button";
import { cn } from "@/lib/utils";

interface AnimatedHoverButtonProps extends React.ComponentProps<typeof Button> {
  children: React.ReactNode;
}

export const AnimatedHoverButton = ({ children, className, ...props }: AnimatedHoverButtonProps) => {
  return (
    <motion.div
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      transition={{ type: "spring", stiffness: 400, damping: 17 }}
    >
      <Button
        className={cn(
          "relative overflow-hidden transition-colors",
          className
        )}
        {...props}
      >
        <motion.div
          className="absolute inset-0 bg-primary/10"
          initial={{ scale: 0, opacity: 0 }}
          whileHover={{ scale: 2, opacity: 1 }}
          transition={{ duration: 0.3 }}
        />
        <motion.span
          className="relative z-10"
          whileHover={{ scale: 1.05 }}
          transition={{ duration: 0.2 }}
        >
          {children}
        </motion.span>
      </Button>
    </motion.div>
  );
}; 