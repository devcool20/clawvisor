import { Navigate, useLocation, useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api, type Agent } from '../api/client'
import { useAuth } from '../hooks/useAuth'

// Decides where `/dashboard` should land. The PRD calls for:
//   - deep-link query params (?action=… with request_id or task_id) →
//     forward to /dashboard/inbox so approvals are never blocked by the
//     agent-gate redirect, even for first-run users
//   - zero agents → /dashboard/quickstart (first-run setup)
//   - ≥1 agent  → /dashboard/home (Overview content)
//
// The agents query is gated on the deep-link check so deep-link redirects
// don't trigger an extra network round-trip.
export default function DashboardIndex() {
  const { currentOrg } = useAuth()
  const location = useLocation()
  const [searchParams] = useSearchParams()

  const action = searchParams.get('action')
  const requestId = searchParams.get('request_id')
  const taskId = searchParams.get('task_id')
  const hasDeepLink = !!action && (!!requestId || !!taskId)

  if (hasDeepLink) {
    return (
      <Navigate
        to={{ pathname: '/dashboard/inbox', search: location.search }}
        replace
      />
    )
  }

  const orgId = currentOrg?.id
  const { data: agents, isLoading } = useQuery({
    queryKey: orgId ? ['org-agents', orgId] : ['agents'],
    queryFn: (): Promise<Agent[]> =>
      orgId ? api.orgs.agents(orgId) : api.agents.list(),
  })

  if (isLoading || !agents) {
    return (
      <div className="p-8 text-sm text-text-tertiary">Loading dashboard…</div>
    )
  }

  if (agents.length === 0) {
    return <Navigate to="/dashboard/quickstart" replace />
  }

  return <Navigate to="/dashboard/home" replace />
}
