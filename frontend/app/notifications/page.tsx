"use client"

import { Bell, CheckCircle2, XCircle, AlertCircle } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

export default function NotificationsPage() {
  const notifications = [
    {
      id: 1,
      title: "Compliance Report Due",
      description: "Your quarterly compliance report is due in 5 days",
      type: "warning",
      date: "2024-03-20",
      read: false,
    },
    {
      id: 2,
      title: "Document Upload Successful",
      description: "Your annual audit document has been successfully uploaded",
      type: "success",
      date: "2024-03-19",
      read: true,
    },
    {
      id: 3,
      title: "Compliance Alert",
      description: "New regulatory requirements have been added to your compliance checklist",
      type: "info",
      date: "2024-03-18",
      read: false,
    },
    {
      id: 4,
      title: "Subscription Update",
      description: "Your subscription has been renewed successfully",
      type: "success",
      date: "2024-03-17",
      read: true,
    },
  ]

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case "success":
        return <CheckCircle2 className="h-5 w-5 text-green-500" />
      case "warning":
        return <AlertCircle className="h-5 w-5 text-yellow-500" />
      case "error":
        return <XCircle className="h-5 w-5 text-red-500" />
      default:
        return <Bell className="h-5 w-5 text-blue-500" />
    }
  }

  return (
    <div className="container py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Notifications</h1>
        <p className="text-muted-foreground">Stay updated with your compliance activities</p>
      </div>

      <div className="space-y-4">
        {notifications.map((notification) => (
          <Card key={notification.id} className={!notification.read ? "border-primary" : ""}>
            <CardContent className="p-6">
              <div className="flex items-start gap-4">
                <div className="mt-1">
                  {getNotificationIcon(notification.type)}
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold">{notification.title}</h3>
                    <span className="text-sm text-muted-foreground">
                      {new Date(notification.date).toLocaleDateString()}
                    </span>
                  </div>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {notification.description}
                  </p>
                </div>
                {!notification.read && (
                  <Button variant="ghost" size="sm">
                    Mark as read
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}

