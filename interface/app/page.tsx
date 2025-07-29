"use client"

import { DashboardHeader } from "@/components/dashboard-header"
import { OverviewCards } from "@/components/overview-cards"
import { AssetAllocation } from "@/components/asset-allocation"
import { PortfolioPerformance } from "@/components/portfolio-performance"
import { RecentTransactions } from "@/components/recent-transactions"
import { GoalsTracker } from "@/components/goals-tracker"
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/app-sidebar"

export default function Home() {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <DashboardHeader title="Dashboard Overview" />
        <div className="flex flex-1 flex-col gap-4 p-4 pt-0">
          <div className="space-y-8">
            <OverviewCards />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <AssetAllocation />
              <PortfolioPerformance />
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
              <div className="xl:col-span-2">
                <GoalsTracker />
              </div>
              <div>
                <RecentTransactions />
              </div>
            </div>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}
