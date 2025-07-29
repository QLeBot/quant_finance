"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export default function FinancialSimulator() {
  const [principal, setPrincipal] = useState(10000);
  const [rate, setRate] = useState(7);
  const [time, setTime] = useState(10);
  const [monthlyContribution, setMonthlyContribution] = useState(500);
  const [compoundingFrequency, setCompoundingFrequency] = useState("monthly");
  const [results, setResults] = useState<any>(null);

  const calculateCompoundInterest = () => {
    const r = rate / 100;
    const n = compoundingFrequency === "monthly" ? 12 : 
              compoundingFrequency === "quarterly" ? 4 : 
              compoundingFrequency === "annually" ? 1 : 12;
    
    // Compound interest formula: A = P(1 + r/n)^(nt)
    const futureValue = principal * Math.pow(1 + r/n, n * time);
    
    // Calculate total contributions
    const totalContributions = principal + (monthlyContribution * 12 * time);
    
    // Calculate interest earned
    const interestEarned = futureValue - totalContributions;
    
    // Generate year-by-year breakdown
    const yearlyBreakdown = [];
    for (let year = 1; year <= time; year++) {
      const yearValue = principal * Math.pow(1 + r/n, n * year) + 
                       (monthlyContribution * 12 * ((Math.pow(1 + r/n, n * year) - 1) / (r/n)));
      yearlyBreakdown.push({
        year,
        value: yearValue,
        contribution: principal + (monthlyContribution * 12 * year),
        interest: yearValue - (principal + (monthlyContribution * 12 * year))
      });
    }

    setResults({
      futureValue,
      totalContributions,
      interestEarned,
      yearlyBreakdown
    });
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 p-4">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Financial Simulator
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            Calculate compound interest, investment growth, and financial projections
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Input Card */}
          <Card>
            <CardHeader>
              <CardTitle>Investment Parameters</CardTitle>
              <CardDescription>
                Enter your investment details to calculate future value
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="principal">Initial Investment</Label>
                <Input
                  id="principal"
                  type="number"
                  value={principal}
                  onChange={(e) => setPrincipal(Number(e.target.value))}
                  placeholder="Enter initial amount"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="rate">Annual Interest Rate (%)</Label>
                <div className="flex items-center space-x-4">
                  <Slider
                    value={[rate]}
                    onValueChange={(value) => setRate(value[0])}
                    max={20}
                    min={0}
                    step={0.1}
                    className="flex-1"
                  />
                  <span className="text-sm font-medium w-12">{rate}%</span>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="time">Investment Period (Years)</Label>
                <div className="flex items-center space-x-4">
                  <Slider
                    value={[time]}
                    onValueChange={(value) => setTime(value[0])}
                    max={50}
                    min={1}
                    step={1}
                    className="flex-1"
                  />
                  <span className="text-sm font-medium w-12">{time} years</span>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="monthlyContribution">Monthly Contribution</Label>
                <Input
                  id="monthlyContribution"
                  type="number"
                  value={monthlyContribution}
                  onChange={(e) => setMonthlyContribution(Number(e.target.value))}
                  placeholder="Enter monthly contribution"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="compounding">Compounding Frequency</Label>
                <Select value={compoundingFrequency} onValueChange={setCompoundingFrequency}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="monthly">Monthly</SelectItem>
                    <SelectItem value="quarterly">Quarterly</SelectItem>
                    <SelectItem value="annually">Annually</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Button onClick={calculateCompoundInterest} className="w-full">
                Calculate Investment Growth
              </Button>
            </CardContent>
          </Card>

          {/* Results Card */}
          <Card>
            <CardHeader>
              <CardTitle>Investment Results</CardTitle>
              <CardDescription>
                Your projected investment growth over time
              </CardDescription>
            </CardHeader>
            <CardContent>
              {results ? (
                <div className="space-y-6">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                      <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                        {formatCurrency(results.futureValue)}
                      </div>
                      <div className="text-sm text-green-600 dark:text-green-400">
                        Future Value
                      </div>
                    </div>
                    <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                      <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                        {formatCurrency(results.interestEarned)}
                      </div>
                      <div className="text-sm text-blue-600 dark:text-blue-400">
                        Interest Earned
                      </div>
                    </div>
                  </div>

                  <div className="text-center p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <div className="text-lg font-semibold text-gray-700 dark:text-gray-300">
                      Total Contributions: {formatCurrency(results.totalContributions)}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <h4 className="font-semibold">Year-by-Year Breakdown</h4>
                    <div className="max-h-64 overflow-y-auto space-y-2">
                      {results.yearlyBreakdown.map((year: any) => (
                        <div key={year.year} className="flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-800 rounded">
                          <span className="font-medium">Year {year.year}</span>
                          <div className="text-right">
                            <div className="font-semibold">{formatCurrency(year.value)}</div>
                            <div className="text-xs text-gray-500">
                              +{formatCurrency(year.interest)} interest
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center text-gray-500 dark:text-gray-400 py-8">
                  Enter your investment parameters and click calculate to see results
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Additional Financial Tools */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Rule of 72</CardTitle>
              <CardDescription>
                Time to double your investment
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                  {(72 / rate).toFixed(1)} years
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  At {rate}% annual return
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Monthly Payment Needed</CardTitle>
              <CardDescription>
                To reach $1M in {time} years
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center">
                <div className="text-3xl font-bold text-orange-600 dark:text-orange-400">
                  {formatCurrency((1000000 - principal * Math.pow(1 + rate/100, time)) / 
                    ((Math.pow(1 + rate/100, time) - 1) / (rate/100)) * (rate/100) / 12)}
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  Monthly contribution needed
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Inflation Impact</CardTitle>
              <CardDescription>
                Real value after 3% inflation
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center">
                <div className="text-3xl font-bold text-red-600 dark:text-red-400">
                  {results ? formatCurrency(results.futureValue / Math.pow(1.03, time)) : '$0'}
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  Inflation-adjusted value
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
