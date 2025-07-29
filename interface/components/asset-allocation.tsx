"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { PieChart, Pie, Cell, ResponsiveContainer } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

const allocationData = [
  { name: "Stocks", value: 45, amount: "$506,250", color: "hsl(var(--chart-1))" },
  { name: "Bonds", value: 25, amount: "$281,250", color: "hsl(var(--chart-2))" },
  { name: "Real Estate", value: 15, amount: "$168,750", color: "hsl(var(--chart-3))" },
  { name: "Commodities", value: 8, amount: "$90,000", color: "hsl(var(--chart-4))" },
  { name: "Cash", value: 7, amount: "$78,750", color: "hsl(var(--chart-5))" },
]

export function AssetAllocation() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Asset Allocation</CardTitle>
        <CardDescription>Distribution of your investment portfolio</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="flex justify-center">
            <ChartContainer
              config={{
                stocks: { label: "Stocks", color: "hsl(var(--chart-1))" },
                bonds: { label: "Bonds", color: "hsl(var(--chart-2))" },
                realEstate: { label: "Real Estate", color: "hsl(var(--chart-3))" },
                commodities: { label: "Commodities", color: "hsl(var(--chart-4))" },
                cash: { label: "Cash", color: "hsl(var(--chart-5))" },
              }}
              className="aspect-square w-full max-w-[250px]"
            >
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={allocationData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {allocationData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <ChartTooltip content={<ChartTooltipContent />} />
                </PieChart>
              </ResponsiveContainer>
            </ChartContainer>
          </div>

          <div className="space-y-4">
            {allocationData.map((item, index) => (
              <div key={index} className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                    <span className="font-medium">{item.name}</span>
                  </div>
                  <div className="text-right">
                    <div className="font-medium">{item.amount}</div>
                    <div className="text-xs text-muted-foreground">{item.value}%</div>
                  </div>
                </div>
                <Progress value={item.value} className="h-2" />
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
