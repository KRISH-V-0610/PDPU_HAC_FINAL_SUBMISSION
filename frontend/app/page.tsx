"use client"

import Link from "next/link"
import { motion } from "framer-motion"
import { ArrowRight, Check, FileText, Shield, Zap, Users, BarChart, Mail, Phone, MessageSquare } from "lucide-react"
import { ThemeToggle } from "@/components/ThemeToggle"
import { AnimatedButton } from "@/components/ui/animated-button"
import { AnimatedCard } from "@/components/ui/animated-card"
// import AnimatedBackground from "@/components/ui/animated-background"
import { useInView } from "react-intersection-observer"
import { AnimatedHoverButton } from "@/components/ui/animated-hover-button"
import Head from "next/head"

const features = [
  {
    title: "Smart File Management",
    description: "Intelligent file organization with automatic categorization and tagging",
    icon: FileText
  },
  {
    title: "Advanced Analytics",
    description: "Comprehensive insights and reporting with customizable dashboards",
    icon: BarChart
  },
  {
    title: "Automated Workflows",
    description: "Streamline your compliance processes with customizable automation rules",
    icon: Zap
  },
  {
    title: "Team Collaboration",
    description: "Real-time collaboration tools for your entire compliance team",
    icon: Users
  },
  {
    title: "Compliance Calendar",
    description: "Never miss a deadline with our intelligent compliance calendar",
    icon: FileText
  },
  {
    title: "Audit Trail",
    description: "Complete visibility into all actions and changes with detailed audit logs",
    icon: Shield
  }
]

const pricingTiers = [
  {
    name: "Free",
    price: "₹0",
    description: "Perfect for small teams getting started",
    features: [
      "5 file uploads per month",
      "Basic analytics dashboard",
      "Email support",
      "Standard file storage",
      "Basic compliance calendar"
    ]
  },
  {
    name: "Premium",
    price: "₹99",
    description: "Best for growing organizations",
    features: [
      "Unlimited file uploads",
      "Advanced analytics & reporting",
      "Priority support (24/7)",
      "Custom compliance workflows",
      "Team collaboration tools",
      "Advanced audit trail",
      "API access",
      "Custom integrations"
    ]
  }
]

export default function HomePage() {
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1
  });

  return (
    <div className="flex min-h-screen flex-col">
      <Head>
        <title>ComplySmart - Simplify Your Compliance Management</title>
      </Head>
      {/* <AnimatedBackground /> */}
      
      <header className="sticky top-0 z-10 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between">
          <motion.div 
            className="flex items-center gap-2"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            <FileText className="h-6 w-6 text-primary" />
            <span className="text-xl font-bold">ComplySmart</span>
          </motion.div>
          <nav className="flex items-center gap-6">
            <Link href="#features" className="text-sm font-medium">
              Features
            </Link>
            <Link href="#pricing" className="text-sm font-medium">
              Pricing
            </Link>
            <ThemeToggle />
            <AnimatedButton asChild>
              <Link href="/upload" className="inline-flex items-center gap-2">
                Upload Files
                {/* <ArrowRight className="h-4 w-4" /> */}
              </Link>
            </AnimatedButton>
          </nav>
        </div>
      </header>

      <main className="flex-1">
        {/* Hero Section */}
        <section className="relative overflow-hidden py-20">
          <div className="container relative z-10">
            <div className="mx-auto max-w-2xl text-center">
              <motion.h1
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="text-4xl font-bold tracking-tight sm:text-6xl"
              >
                Simplify Your{" "}
                <span className="text-primary">Compliance Management</span>
              </motion.h1>
              <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
                className="mt-6 text-lg text-muted-foreground"
              >
                Upload, track, and manage your compliance reports with ease.
                Our platform helps you stay compliant and organized.
              </motion.p>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="mt-10 flex items-center justify-center gap-4"
              >
                <AnimatedButton asChild>
                  <Link href="/auth" className="inline-flex items-center gap-2">
                    Get Started
                    {/* <ArrowRight className="h-4 w-4" /> */}
                  </Link>
                </AnimatedButton>
              </motion.div>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section id="features" className="py-20" ref={ref}>
          <div className="container">
            <motion.div 
              className="mx-auto max-w-2xl text-center"
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.5 }}
            >
              <h2 className="text-3xl font-bold tracking-tight">
                Powerful Features for Modern Compliance
              </h2>
              <p className="mt-4 text-muted-foreground">
                Our platform combines cutting-edge technology with intuitive design to make compliance management effortless.
              </p>
            </motion.div>
            <div className="mt-16 grid gap-8 sm:grid-cols-2 lg:grid-cols-3">
              {features.map((feature, index) => (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={inView ? { opacity: 1, y: 0 } : {}}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                >
                  <AnimatedCard 
                    className="h-full flex flex-col group"
                    as={motion.div}
                    whileHover={{ 
                      scale: 1.02,
                      y: -5,
                      transition: { 
                        type: "spring",
                        stiffness: 300,
                        damping: 15
                      }
                    }}
                  >
                    <div className="relative">
                      <feature.icon className="h-10 w-10 text-primary transition-transform duration-300 group-hover:scale-110" />
                      <div className="absolute -bottom-1 left-0 h-0.5 w-0 bg-primary transition-all duration-300 group-hover:w-full" />
                    </div>
                    <h3 className="mt-4 text-lg font-medium">
                      {feature.title}
                    </h3>
                    <p className="mt-2 text-muted-foreground flex-grow">
                      {feature.description}
                    </p>
                  </AnimatedCard>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Pricing Section */}
        <section id="pricing" className="py-20">
          <div className="container">
            <motion.div 
              className="mx-auto max-w-2xl text-center"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              viewport={{ once: true }}
            >
              <h2 className="text-3xl font-bold tracking-tight">
                Simple, Transparent Pricing
              </h2>
              <p className="mt-4 text-muted-foreground">
                Choose the plan that best fits your organization's needs
              </p>
            </motion.div>
            <div className="mt-16 grid gap-8 sm:grid-cols-2">
              {pricingTiers.map((tier, index) => (
                <motion.div
                  key={tier.name}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <AnimatedCard 
                    className="h-full flex flex-col group"
                    as={motion.div}
                    whileHover={{ 
                      scale: 1.02,
                      y: -5,
                      transition: { 
                        type: "spring",
                        stiffness: 300,
                        damping: 15
                      }
                    }}
                  >
                    <div className="flex-grow">
                      <div className="relative">
                        <h3 className="text-2xl font-bold">{tier.name}</h3>
                        <div className="absolute -bottom-1 left-0 h-0.5 w-0 bg-primary transition-all duration-300 group-hover:w-full" />
                      </div>
                      <p className="mt-2 text-4xl font-bold">{tier.price}</p>
                      <p className="mt-2 text-muted-foreground">
                        {tier.description}
                      </p>
                      <ul className="mt-8 space-y-4">
                        {tier.features.map((feature) => (
                          <li key={feature} className="flex items-center gap-2">
                            <Check className="h-4 w-4 text-primary transition-transform duration-300 group-hover:scale-110" />
                            {feature}
                          </li>
                        ))}
                      </ul>
                    </div>
                    <div className="mt-8">
                      <AnimatedButton
                        className="w-full"
                        variant={tier.name === "Premium" ? "default" : "outline"}
                        asChild
                      >
                        <Link href="/auth" className="inline-flex items-center justify-center gap-2">
                          {tier.name === "Free" ? "Get Started" : "Upgrade Now"}
                          <ArrowRight className="h-4 w-4 transition-transform duration-300 group-hover:translate-x-1" />
                        </Link>
                      </AnimatedButton>
                    </div>
                  </AnimatedCard>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Contact Section */}
        <section id="contact" className="py-20">
          <div className="container">
            <motion.div 
              className="mx-auto max-w-2xl text-center"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              viewport={{ once: true }}
            >
              <h2 className="text-3xl font-bold tracking-tight">
                Get in Touch
              </h2>
              <p className="mt-4 text-muted-foreground">
                Have questions? We'd love to hear from you. Send us a message and we'll respond as soon as possible.
              </p>
            </motion.div>
            <div className="mt-16 grid gap-8 sm:grid-cols-2 lg:grid-cols-3">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
                viewport={{ once: true }}
              >
                <AnimatedCard className="h-full">
                  <div className="flex flex-col items-center text-center">
                    <div className="rounded-full bg-primary/10 p-4">
                      <Mail className="h-6 w-6 text-primary" />
                    </div>
                    <h3 className="mt-4 text-lg font-medium">Email Us</h3>
                    <p className="mt-2 text-muted-foreground">
                      support@compliancetrack.com
                    </p>
                  </div>
                </AnimatedCard>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                viewport={{ once: true }}
              >
                <AnimatedCard className="h-full">
                  <div className="flex flex-col items-center text-center">
                    <div className="rounded-full bg-primary/10 p-4">
                      <Phone className="h-6 w-6 text-primary" />
                    </div>
                    <h3 className="mt-4 text-lg font-medium">Call Us</h3>
                    <p className="mt-2 text-muted-foreground">
                      +1 (555) 123-4567
                    </p>
                  </div>
                </AnimatedCard>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.3 }}
                viewport={{ once: true }}
              >
                <AnimatedCard className="h-full">
                  <div className="flex flex-col items-center text-center">
                    <div className="rounded-full bg-primary/10 p-4">
                      <MessageSquare className="h-6 w-6 text-primary" />
                    </div>
                    <h3 className="mt-4 text-lg font-medium">Live Chat</h3>
                    <p className="mt-2 text-muted-foreground">
                      Available 24/7 for instant support
                    </p>
                  </div>
                </AnimatedCard>
              </motion.div>
            </div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              viewport={{ once: true }}
              className="mt-12 text-center"
            >
              <AnimatedButton asChild>
                <Link href="/contact" className="inline-flex items-center gap-2">
                  Contact Support
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </AnimatedButton>
            </motion.div>
          </div>
        </section>
      </main>

      <footer className="border-t py-6">
        <div className="container flex flex-col items-center justify-between gap-4 md:flex-row">
          <p className="text-center text-sm text-muted-foreground">
            &copy; {new Date().getFullYear()} ComplySmart. All rights reserved.
          </p>
          <p className="text-center text-sm text-muted-foreground">
            Designed for efficient compliance management
          </p>
        </div>
      </footer>
    </div>
  )
}

