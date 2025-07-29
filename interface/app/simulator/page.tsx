"use client"

import { DashboardHeader } from "@/components/dashboard-header"
import { InvestmentSimulator } from "@/components/investment-simulator"
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/app-sidebar"

export default function SimulatorPage() {
  const breadcrumbs = [{ label: "Tools & Simulators" }, { label: "Investment Simulator" }]

  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <DashboardHeader title="Investment Simulator" breadcrumbs={breadcrumbs} />
        <div className="flex flex-1 flex-col gap-4 p-4 pt-0">
          <div className="space-y-8">
            <InvestmentSimulator />

            {/* Additional simulator tools can go here */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="p-6 border rounded-lg">
                <h3 className="font-semibold mb-2">Retirement Calculator</h3>
                <p className="text-sm text-muted-foreground mb-4">Calculate how much you need to save for retirement</p>
                <button className="text-sm text-primary hover:underline">Coming Soon</button>
              </div>

              <div className="p-6 border rounded-lg">
                <h3 className="font-semibold mb-2">Loan Calculator</h3>
                <p className="text-sm text-muted-foreground mb-4">Calculate monthly payments and interest for loans</p>
                <button className="text-sm text-primary hover:underline">Coming Soon</button>
              </div>

              <div className="p-6 border rounded-lg">
                <h3 className="font-semibold mb-2">Tax Calculator</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Estimate your tax liability and optimize strategies
                </p>
                <button className="text-sm text-primary hover:underline">Coming Soon</button>
              </div>
            </div>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}
