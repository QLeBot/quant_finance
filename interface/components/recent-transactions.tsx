"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ArrowUpRight, ArrowDownRight, History } from "lucide-react"

const transactions = [
  {
    type: "buy",
    asset: "AAPL",
    amount: "$2,500",
    date: "2 hours ago",
    status: "completed",
  },
  {
    type: "sell",
    asset: "TSLA",
    amount: "$1,800",
    date: "1 day ago",
    status: "completed",
  },
  {
    type: "dividend",
    asset: "VTI",
    amount: "$125",
    date: "3 days ago",
    status: "completed",
  },
  {
    type: "buy",
    asset: "BTC",
    amount: "$500",
    date: "1 week ago",
    status: "pending",
  },
]

export function RecentTransactions() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <History className="h-5 w-5" />
          <span>Recent Activity</span>
        </CardTitle>
        <CardDescription>Your latest investment transactions</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {transactions.map((transaction, index) => (
          <div key={index} className="flex items-center justify-between p-3 rounded-lg border">
            <div className="flex items-center space-x-3">
              <div
                className={`p-2 rounded-full ${
                  transaction.type === "buy"
                    ? "bg-green-100 text-green-600"
                    : transaction.type === "sell"
                      ? "bg-red-100 text-red-600"
                      : "bg-blue-100 text-blue-600"
                }`}
              >
                {transaction.type === "buy" ? (
                  <ArrowUpRight className="h-4 w-4" />
                ) : transaction.type === "sell" ? (
                  <ArrowDownRight className="h-4 w-4" />
                ) : (
                  <ArrowDownRight className="h-4 w-4" />
                )}
              </div>
              <div>
                <div className="font-medium text-sm">
                  {transaction.type === "buy" ? "Bought" : transaction.type === "sell" ? "Sold" : "Dividend"}{" "}
                  {transaction.asset}
                </div>
                <div className="text-xs text-muted-foreground">{transaction.date}</div>
              </div>
            </div>
            <div className="text-right">
              <div className="font-medium text-sm">{transaction.amount}</div>
              <Badge variant={transaction.status === "completed" ? "default" : "secondary"} className="text-xs">
                {transaction.status}
              </Badge>
            </div>
          </div>
        ))}

        <Button variant="outline" className="w-full bg-transparent">
          View All Transactions
        </Button>
      </CardContent>
    </Card>
  )
}
