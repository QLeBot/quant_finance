"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, BarChart, Bar } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

const growthData = [
  { month: "Jan", value: 850000 },
  { month: "Feb", value: 865000 },
  { month: "Mar", value: 882000 },
  { month: "Apr", value: 895000 },
  { month: "May", value: 912000 },
  { month: "Jun", value: 925000 },
  { month: "Jul", value: 940000 },
  { month: "Aug", value: 955000 },
  { month: "Sep", value: 968000 },
  { month: "Oct", value: 985000 },
  { month: "Nov", value: 1002000 },
  { month: "Dec", value: 1025000 },
]

const returnsData = [
  { month: "Jan", return: 1.8 },
  { month: "Feb", return: 2.1 },
  { month: "Mar", return: -0.5 },
  { month: "Apr", return: 1.2 },
  { month: "May", return: 2.4 },
  { month: "Jun", return: 0.8 },
  { month: "Jul", return: 1.6 },
  { month: "Aug", return: -1.2 },
  { month: "Sep", return: 1.9 },
  { month: "Oct", return: 0.7 },
  { month: "Nov", return: 1.4 },
  { month: "Dec", return: 2.2 },
]

export function PortfolioPerformance() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Portfolio Performance</CardTitle>
        <CardDescription>Track your investment performance over time</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="growth" className="space-y-4">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="growth">Growth</TabsTrigger>
            <TabsTrigger value="returns">Monthly Returns</TabsTrigger>
          </TabsList>

          <TabsContent value="growth" className="space-y-4">
            <ChartContainer
              config={{
                value: { label: "Portfolio Value", color: "hsl(var(--chart-1))" },
              }}
              className="aspect-[4/3]"
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={growthData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`} />
                  <ChartTooltip
                    content={<ChartTooltipContent />}
                    formatter={(value) => [`$${Number(value).toLocaleString()}`, "Portfolio Value"]}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="var(--color-value)"
                    strokeWidth={2}
                    dot={{ fill: "var(--color-value)" }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold text-green-600">+18.5%</div>
                <div className="text-xs text-muted-foreground">YTD Return</div>
              </div>
              <div>
                <div className="text-2xl font-bold">+12.3%</div>
                <div className="text-xs text-muted-foreground">1Y Return</div>
              </div>
              <div>
                <div className="text-2xl font-bold">+8.7%</div>
                <div className="text-xs text-muted-foreground">3Y Avg</div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="returns" className="space-y-4">
            <ChartContainer
              config={{
                return: { label: "Monthly Return", color: "hsl(var(--chart-2))" },
              }}
              className="aspect-[4/3]"
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={returnsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis tickFormatter={(value) => `${value}%`} />
                  <ChartTooltip
                    content={<ChartTooltipContent />}
                    formatter={(value) => [`${value}%`, "Monthly Return"]}
                  />
                  <Bar dataKey="return" fill="var(--color-return)" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
            <div className="grid grid-cols-2 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold text-green-600">+2.4%</div>
                <div className="text-xs text-muted-foreground">Best Month</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-red-600">-1.2%</div>
                <div className="text-xs text-muted-foreground">Worst Month</div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
