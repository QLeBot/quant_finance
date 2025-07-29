"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { 
  Banknote, 
  TrendingUp, 
  Shield, 
  RefreshCw, 
  ExternalLink,
  AlertTriangle,
  CheckCircle,
  DollarSign,
  PieChart,
  Activity
} from "lucide-react"

interface PowensAccount {
  id: string
  name: string
  type: string
  balance: number
  available_balance: number
  institution_name: string
  last_sync_date: string
}

interface PortfolioSummary {
  total_balance: number
  total_accounts: number
  account_types: Record<string, number>
  institutions: Record<string, number>
  last_sync: string | null
}

interface PortfolioMetrics {
  diversification_score: number
  herfindahl_index: number
  average_account_balance: number
  largest_account_share: number
}

interface RiskMetrics {
  institution_concentration: number
  type_concentration: number
  sync_risk: number
  number_of_institutions: number
  diversification_score: number
}

export function PowensIntegration() {
  const [accounts, setAccounts] = useState<PowensAccount[]>([])
  const [summary, setSummary] = useState<PortfolioSummary | null>(null)
  const [metrics, setMetrics] = useState<PortfolioMetrics | null>(null)
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [webviewUrl, setWebviewUrl] = useState<string | null>(null)

  // Mock data for demonstration - replace with actual API calls
  useEffect(() => {
    // Simulate loading data
    setLoading(true)
    
    setTimeout(() => {
      // Mock data
      const mockAccounts: PowensAccount[] = [
        {
          id: "1",
          name: "Main Checking",
          type: "checking",
          balance: 15420.50,
          available_balance: 15420.50,
          institution_name: "Chase Bank",
          last_sync_date: "2024-01-15T10:30:00Z"
        },
        {
          id: "2", 
          name: "Savings Account",
          type: "savings",
          balance: 45230.75,
          available_balance: 45230.75,
          institution_name: "Chase Bank",
          last_sync_date: "2024-01-15T10:30:00Z"
        },
        {
          id: "3",
          name: "Investment Portfolio",
          type: "investment",
          balance: 125430.25,
          available_balance: 125430.25,
          institution_name: "Fidelity",
          last_sync_date: "2024-01-14T15:45:00Z"
        }
      ]

      const mockSummary: PortfolioSummary = {
        total_balance: 186081.50,
        total_accounts: 3,
        account_types: { checking: 1, savings: 1, investment: 1 },
        institutions: { "Chase Bank": 2, "Fidelity": 1 },
        last_sync: "2024-01-15T10:30:00Z"
      }

      const mockMetrics: PortfolioMetrics = {
        diversification_score: 0.67,
        herfindahl_index: 0.33,
        average_account_balance: 62027.17,
        largest_account_share: 0.67
      }

      const mockRiskMetrics: RiskMetrics = {
        institution_concentration: 0.56,
        type_concentration: 0.33,
        sync_risk: 0.0,
        number_of_institutions: 2,
        diversification_score: 0.67
      }

      setAccounts(mockAccounts)
      setSummary(mockSummary)
      setMetrics(mockMetrics)
      setRiskMetrics(mockRiskMetrics)
      setLoading(false)
    }, 1000)
  }, [])

  const handleSyncAccounts = async () => {
    setLoading(true)
    setError(null)
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      // Update last sync time
      if (summary) {
        setSummary({
          ...summary,
          last_sync: new Date().toISOString()
        })
      }
      
      setLoading(false)
    } catch (err) {
      setError("Failed to sync accounts")
      setLoading(false)
    }
  }

  const handleConnectAccounts = () => {
    // In a real implementation, this would redirect to Powens webview
    const mockWebviewUrl = "https://webview.powens.com/auth?user_id=example&redirect_url=https://your-app.com/callback"
    setWebviewUrl(mockWebviewUrl)
  }

  const getAccountTypeIcon = (type: string) => {
    switch (type) {
      case "checking":
        return <Banknote className="h-4 w-4" />
      case "savings":
        return <TrendingUp className="h-4 w-4" />
      case "investment":
        return <PieChart className="h-4 w-4" />
      default:
        return <DollarSign className="h-4 w-4" />
    }
  }

  const getSyncStatus = (lastSync: string) => {
    const lastSyncDate = new Date(lastSync)
    const now = new Date()
    const hoursSinceSync = (now.getTime() - lastSyncDate.getTime()) / (1000 * 60 * 60)
    
    if (hoursSinceSync < 24) {
      return { status: "success", icon: <CheckCircle className="h-3 w-3" />, text: "Recent" }
    } else if (hoursSinceSync < 168) { // 7 days
      return { status: "warning", icon: <AlertTriangle className="h-3 w-3" />, text: "Stale" }
    } else {
      return { status: "error", icon: <AlertTriangle className="h-3 w-3" />, text: "Outdated" }
    }
  }

  if (loading && !accounts.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Powens Integration
          </CardTitle>
          <CardDescription>Loading financial data...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <RefreshCw className="h-8 w-8 animate-spin" />
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Financial Accounts</h2>
          <p className="text-muted-foreground">
            Connected accounts and portfolio analysis powered by Powens
          </p>
        </div>
        <div className="flex gap-2">
          <Button 
            variant="outline" 
            onClick={handleConnectAccounts}
            disabled={loading}
          >
            <ExternalLink className="h-4 w-4 mr-2" />
            Connect Accounts
          </Button>
          <Button 
            onClick={handleSyncAccounts}
            disabled={loading}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Sync Accounts
          </Button>
        </div>
      </div>

      {error && (
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <div className="flex items-center gap-2 text-destructive">
              <AlertTriangle className="h-4 w-4" />
              {error}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Portfolio Summary */}
      {summary && (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Balance</CardTitle>
              <DollarSign className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                ${summary.total_balance.toLocaleString()}
              </div>
              <p className="text-xs text-muted-foreground">
                Across {summary.total_accounts} accounts
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Diversification</CardTitle>
              <Shield className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {metrics ? `${(metrics.diversification_score * 100).toFixed(0)}%` : "N/A"}
              </div>
              <p className="text-xs text-muted-foreground">
                Portfolio diversification score
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Institutions</CardTitle>
              <Banknote className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {Object.keys(summary.institutions).length}
              </div>
              <p className="text-xs text-muted-foreground">
                Financial institutions
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Last Sync</CardTitle>
              <RefreshCw className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {summary.last_sync ? new Date(summary.last_sync).toLocaleDateString() : "Never"}
              </div>
              <p className="text-xs text-muted-foreground">
                Account synchronization
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Accounts and Analysis */}
      <Tabs defaultValue="accounts" className="space-y-4">
        <TabsList>
          <TabsTrigger value="accounts">Accounts</TabsTrigger>
          <TabsTrigger value="analysis">Risk Analysis</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
        </TabsList>

        <TabsContent value="accounts" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Connected Accounts</CardTitle>
              <CardDescription>
                Your financial accounts and their current status
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {accounts.map((account) => {
                  const syncStatus = getSyncStatus(account.last_sync_date)
                  return (
                    <div key={account.id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center gap-3">
                        {getAccountTypeIcon(account.type)}
                        <div>
                          <div className="font-medium">{account.name}</div>
                          <div className="text-sm text-muted-foreground">
                            {account.institution_name} • {account.type}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-right">
                          <div className="font-medium">
                            ${account.balance.toLocaleString()}
                          </div>
                          <div className="text-sm text-muted-foreground">
                            Available: ${account.available_balance.toLocaleString()}
                          </div>
                        </div>
                        <Badge variant={syncStatus.status as any}>
                          {syncStatus.icon}
                          <span className="ml-1">{syncStatus.text}</span>
                        </Badge>
                      </div>
                    </div>
                  )
                })}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analysis" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Risk Analysis</CardTitle>
              <CardDescription>
                Portfolio risk metrics and concentration analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              {riskMetrics && (
                <div className="space-y-6">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">Institution Concentration</span>
                        <span className="text-sm text-muted-foreground">
                          {(riskMetrics.institution_concentration * 100).toFixed(1)}%
                        </span>
                      </div>
                      <Progress value={riskMetrics.institution_concentration * 100} />
                    </div>
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">Type Concentration</span>
                        <span className="text-sm text-muted-foreground">
                          {(riskMetrics.type_concentration * 100).toFixed(1)}%
                        </span>
                      </div>
                      <Progress value={riskMetrics.type_concentration * 100} />
                    </div>
                  </div>
                  
                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">
                        {(riskMetrics.diversification_score * 100).toFixed(0)}%
                      </div>
                      <div className="text-sm text-muted-foreground">Diversification Score</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold">
                        {riskMetrics.number_of_institutions}
                      </div>
                      <div className="text-sm text-muted-foreground">Institutions</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-orange-600">
                        {(riskMetrics.sync_risk * 100).toFixed(0)}%
                      </div>
                      <div className="text-sm text-muted-foreground">Sync Risk</div>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Performance Overview</CardTitle>
              <CardDescription>
                Portfolio performance and cash flow analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-muted-foreground">
                <TrendingUp className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Performance data will be available once accounts are connected</p>
                <Button 
                  variant="outline" 
                  className="mt-4"
                  onClick={handleConnectAccounts}
                >
                  Connect Accounts to View Performance
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Webview Modal */}
      {webviewUrl && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold mb-4">Connect Your Accounts</h3>
            <p className="text-muted-foreground mb-4">
              You'll be redirected to Powens to securely connect your financial accounts.
            </p>
            <div className="flex gap-2">
              <Button 
                onClick={() => window.open(webviewUrl, '_blank')}
                className="flex-1"
              >
                Continue to Powens
              </Button>
              <Button 
                variant="outline" 
                onClick={() => setWebviewUrl(null)}
              >
                Cancel
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
} 