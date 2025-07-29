"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Button } from "@/components/ui/button"
import { Target, Plus } from "lucide-react"

const goals = [
  {
    name: "Emergency Fund",
    target: 50000,
    current: 35000,
    deadline: "Dec 2024",
  },
  {
    name: "House Down Payment",
    target: 100000,
    current: 65000,
    deadline: "Jun 2025",
  },
  {
    name: "Retirement Fund",
    target: 1000000,
    current: 450000,
    deadline: "2045",
  },
]

export function GoalsTracker() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Target className="h-5 w-5" />
          <span>Financial Goals</span>
        </CardTitle>
        <CardDescription>Track progress towards your financial objectives</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {goals.map((goal, index) => {
          const progress = (goal.current / goal.target) * 100
          return (
            <div key={index} className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium">{goal.name}</span>
                <span className="text-muted-foreground">{goal.deadline}</span>
              </div>
              <Progress value={progress} className="h-2" />
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>
                  ${goal.current.toLocaleString()} / ${goal.target.toLocaleString()}
                </span>
                <span>{progress.toFixed(1)}%</span>
              </div>
            </div>
          )
        })}

        <Button variant="outline" className="w-full mt-4 bg-transparent">
          <Plus className="mr-2 h-4 w-4" />
          Add New Goal
        </Button>
      </CardContent>
    </Card>
  )
}
