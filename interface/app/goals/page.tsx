"use client"

import { DashboardHeader } from "@/components/dashboard-header"
import { GoalsTracker } from "@/components/goals-tracker"
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/app-sidebar"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Plus, Target, Calendar, DollarSign } from "lucide-react"

export default function GoalsPage() {
  const breadcrumbs = [{ label: "Tools & Simulators" }, { label: "Goal Tracker" }]

  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <DashboardHeader title="Financial Goals" breadcrumbs={breadcrumbs} />
        <div className="flex flex-1 flex-col gap-4 p-4 pt-0">
          <div className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Active Goals</CardTitle>
                  <Target className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">3</div>
                  <p className="text-xs text-muted-foreground">Currently tracking</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Target</CardTitle>
                  <DollarSign className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">$1.15M</div>
                  <p className="text-xs text-muted-foreground">Across all goals</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Next Deadline</CardTitle>
                  <Calendar className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">Dec 2024</div>
                  <p className="text-xs text-muted-foreground">Emergency Fund</p>
                </CardContent>
              </Card>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <GoalsTracker />

              <Card>
                <CardHeader>
                  <CardTitle>Goal Insights</CardTitle>
                  <CardDescription>Tips to reach your financial goals faster</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="p-4 bg-muted rounded-lg">
                    <h4 className="font-medium mb-2">💡 Tip: Emergency Fund</h4>
                    <p className="text-sm text-muted-foreground">
                      You're 70% towards your emergency fund goal. Consider increasing your monthly savings by $200 to
                      reach it 2 months earlier.
                    </p>
                  </div>

                  <div className="p-4 bg-muted rounded-lg">
                    <h4 className="font-medium mb-2">📈 Opportunity: House Down Payment</h4>
                    <p className="text-sm text-muted-foreground">
                      Based on your current savings rate, you could reach your house down payment goal by April 2025
                      instead of June.
                    </p>
                  </div>

                  <Button className="w-full">
                    <Plus className="mr-2 h-4 w-4" />
                    Create New Goal
                  </Button>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}
