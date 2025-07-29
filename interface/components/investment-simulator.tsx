"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Calculator, TrendingUp } from "lucide-react"

export function InvestmentSimulator() {
  const [initialAmount, setInitialAmount] = useState(10000)
  const [monthlyContribution, setMonthlyContribution] = useState(500)
  const [annualReturn, setAnnualReturn] = useState([7])
  const [timeHorizon, setTimeHorizon] = useState([20])

  const calculateFutureValue = () => {
    const monthlyRate = annualReturn[0] / 100 / 12
    const months = timeHorizon[0] * 12

    // Future value of initial investment
    const futureValueInitial = initialAmount * Math.pow(1 + monthlyRate, months)

    // Future value of monthly contributions (annuity)
    const futureValueContributions = monthlyContribution * ((Math.pow(1 + monthlyRate, months) - 1) / monthlyRate)

    return futureValueInitial + futureValueContributions
  }

  const futureValue = calculateFutureValue()
  const totalContributions = initialAmount + monthlyContribution * timeHorizon[0] * 12
  const totalGains = futureValue - totalContributions

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Calculator className="h-5 w-5" />
          <span>Investment Simulator</span>
        </CardTitle>
        <CardDescription>Project your investment growth over time</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="calculator" className="space-y-6">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="calculator">Calculator</TabsTrigger>
            <TabsTrigger value="scenarios">Scenarios</TabsTrigger>
          </TabsList>

          <TabsContent value="calculator" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="initial">Initial Investment</Label>
                  <Input
                    id="initial"
                    type="number"
                    value={initialAmount}
                    onChange={(e) => setInitialAmount(Number(e.target.value))}
                    className="text-right"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="monthly">Monthly Contribution</Label>
                  <Input
                    id="monthly"
                    type="number"
                    value={monthlyContribution}
                    onChange={(e) => setMonthlyContribution(Number(e.target.value))}
                    className="text-right"
                  />
                </div>

                <div className="space-y-3">
                  <Label>Annual Return: {annualReturn[0]}%</Label>
                  <Slider
                    value={annualReturn}
                    onValueChange={setAnnualReturn}
                    max={15}
                    min={1}
                    step={0.5}
                    className="w-full"
                  />
                </div>

                <div className="space-y-3">
                  <Label>Time Horizon: {timeHorizon[0]} years</Label>
                  <Slider
                    value={timeHorizon}
                    onValueChange={setTimeHorizon}
                    max={40}
                    min={1}
                    step={1}
                    className="w-full"
                  />
                </div>
              </div>

              <div className="space-y-4">
                <div className="p-6 bg-muted rounded-lg space-y-4">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-primary">
                      ${futureValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}
                    </div>
                    <div className="text-sm text-muted-foreground">Projected Value</div>
                  </div>

                  <div className="grid grid-cols-2 gap-4 text-center">
                    <div>
                      <div className="text-lg font-semibold">${totalContributions.toLocaleString("en-US")}</div>
                      <div className="text-xs text-muted-foreground">Total Invested</div>
                    </div>
                    <div>
                      <div className="text-lg font-semibold text-green-600">
                        ${totalGains.toLocaleString("en-US", { maximumFractionDigits: 0 })}
                      </div>
                      <div className="text-xs text-muted-foreground">Total Gains</div>
                    </div>
                  </div>
                </div>

                <Button className="w-full">
                  <TrendingUp className="mr-2 h-4 w-4" />
                  Save Simulation
                </Button>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="scenarios" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Conservative</CardTitle>
                  <CardDescription className="text-xs">5% annual return</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="text-xl font-bold">$284,500</div>
                  <div className="text-xs text-muted-foreground">20-year projection</div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Moderate</CardTitle>
                  <CardDescription className="text-xs">7% annual return</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="text-xl font-bold">$372,800</div>
                  <div className="text-xs text-muted-foreground">20-year projection</div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Aggressive</CardTitle>
                  <CardDescription className="text-xs">10% annual return</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="text-xl font-bold">$525,400</div>
                  <div className="text-xs text-muted-foreground">20-year projection</div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
