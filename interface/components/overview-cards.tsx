"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { TrendingUp, TrendingDown, DollarSign, PiggyBank, Wallet } from "lucide-react"

const overviewData = [
  {
    title: "Net Worth",
    value: "$847,250",
    change: "+12.5%",
    changeType: "positive" as const,
    icon: DollarSign,
    description: "Total assets minus liabilities",
  },
  {
    title: "Gross Worth",
    value: "$1,125,000",
    change: "+8.2%",
    changeType: "positive" as const,
    icon: Wallet,
    description: "Total value of all assets",
  },
  {
    title: "Monthly Savings",
    value: "$4,850",
    change: "+15.3%",
    changeType: "positive" as const,
    icon: PiggyBank,
    description: "Average monthly savings",
  },
  {
    title: "Investment Returns",
    value: "$12,450",
    change: "-2.1%",
    changeType: "negative" as const,
    icon: TrendingUp,
    description: "This month's returns",
  },
]

export function OverviewCards() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {overviewData.map((item, index) => (
        <Card key={index} className="relative overflow-hidden">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">{item.title}</CardTitle>
            <item.icon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{item.value}</div>
            <div className="flex items-center space-x-2 text-xs text-muted-foreground">
              <div
                className={`flex items-center ${item.changeType === "positive" ? "text-green-600" : "text-red-600"}`}
              >
                {item.changeType === "positive" ? (
                  <TrendingUp className="h-3 w-3 mr-1" />
                ) : (
                  <TrendingDown className="h-3 w-3 mr-1" />
                )}
                {item.change}
              </div>
              <span>from last month</span>
            </div>
            <CardDescription className="mt-2">{item.description}</CardDescription>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
