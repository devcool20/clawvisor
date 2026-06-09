import GetStarted from './GetStarted'

// Stub: the Quickstart route is the new home for setup content. Until the
// dedicated Quickstart layout is built it renders GetStarted, which already
// covers the feature-aware setup checklist + how-it-works copy. The route
// name is the stable contract — `/dashboard/quickstart` is what the nav,
// the agent-gate redirect, and Telegram links target.
export default function Quickstart() {
  return <GetStarted />
}
