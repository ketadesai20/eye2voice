# Eye2Voice Developer Guide

A comprehensive guide to current Eye2Voice codebase — an accessibility-focused gaze-to-speech communication web app.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Tech Stack & Dependencies](#tech-stack--dependencies)
3. [Getting Started](#getting-started)
4. [Project Structure](#project-structure)
5. [Screen Flow & Routing](#screen-flow--routing)
6. [Component Breakdown](#component-breakdown)
7. [Response Tree Data Model](#response-tree-data-model)
8. [Design System](#design-system)
9. [Design Reference Files](#design-reference-files)
10. [State Management](#state-management)
11. [Testing](#testing)
12. [Common Tasks](#common-tasks)

---

## Project Overview

Eye2Voice helps users who cannot speak communicate by navigating a response tree using eye tracking (gaze detection). Users look at quadrants on screen — top, left, right, bottom — to drill through categories like "Pain > Physical Pain > Head > Severe" and construct a response.

The app is currently a **UI skeleton** with full click-through navigation. The ML gaze detection model has not yet been integrated — clicks simulate what gaze selection will eventually do.

---

## Tech Stack & Dependencies

### Core Framework

| Package | Version | Purpose |
|---------|---------|---------|
| `react` | 19.2.4 | UI component library |
| `react-dom` | 19.2.4 | React DOM renderer |
| `react-router-dom` | 7.13.0 | Client-side routing (`BrowserRouter`, `Routes`, `Route`, `useNavigate`, `useParams`) |
| `react-scripts` | 5.0.1 | Create React App toolchain (Webpack, Babel, ESLint, dev server) |

### Testing

| Package | Version | Purpose |
|---------|---------|---------|
| `@testing-library/react` | 16.3.2 | React component testing utilities |
| `@testing-library/jest-dom` | 6.6.3 | Custom Jest matchers for DOM assertions (e.g., `toBeInTheDocument()`) |
| `@testing-library/user-event` | 13.5.0 | Simulates user interactions in tests |

### Other

| Package | Version | Purpose |
|---------|---------|---------|
| `web-vitals` | 2.1.4 | Performance metrics reporting (CRA default) |

### External Resources (loaded via CDN in `public/index.html`)

- **Google Fonts — Josefin Sans** (weights 300, 400, 500, 600): The app's primary typeface, loaded via `<link>` tags in the HTML head.

---

## Getting Started

All commands run from the `eye2voice-ui/` directory:

```bash
cd eye2voice-ui

# Install dependencies
npm install

# Start development server (http://localhost:3000)
npm start

# Run tests in watch mode
npm test

# Run tests with coverage report
npm test -- --coverage

# Run a single test file
npm test -- --testPathPattern=App.test.js

# Production build (outputs to /build)
npm run build
```

The dev server supports hot reloading — saved changes appear instantly in the browser.

---

## Project Structure

```
eye2voice-ui/
├── public/
│   ├── index.html              # HTML template, Google Fonts links, meta tags
│   ├── eye2voice_logo.png      # Main logo used on WelcomeScreen
│   ├── logo_img.png            # Alternative logo asset
│   ├── favicon.ico             # Browser tab icon
│   ├── logo192.png             # PWA icon (192x192)
│   ├── logo512.png             # PWA icon (512x512)
│   ├── manifest.json           # PWA manifest
│   └── robots.txt              # Search engine crawling rules
│
├── src/
│   ├── index.js                # React entry point — renders <App /> into DOM
│   ├── index.css               # Global body/font styles
│   ├── App.js                  # Router setup — defines all routes
│   ├── App.css                 # Global app container styles
│   │
│   ├── WelcomeScreen.js        # Landing screen with swipe-up gesture
│   ├── WelcomeScreen.css       # Welcome screen styles + bounce animation
│   │
│   ├── MenuScreen.js           # "Demo" or "Consent & Talk" choice
│   ├── MenuScreen.css          # Menu screen styles
│   │
│   ├── ConsentScreen.js        # Camera/data consent checkboxes
│   ├── ConsentScreen.css       # Consent screen styles
│   │
│   ├── HomePage.js             # Root quadrant page (Respond/Need/Ask/Pain)
│   ├── ResponsePage.js         # Dynamic response tree navigation page
│   ├── QuadrantPage.css        # Shared styles for HomePage + ResponsePage
│   │
│   ├── responseData.js         # Response tree data + helper functions
│   │
│   ├── App.test.js             # Placeholder test
│   ├── setupTests.js           # Jest-DOM setup
│   ├── reportWebVitals.js      # CRA performance reporting
│   └── logo.svg                # Default CRA React logo (unused)
│
├── data/
│   └── responses.csv           # Source of truth for the response tree
│
├── package.json                # Dependencies and scripts
├── CLAUDE.md                   # AI assistant instructions for this project
└── DEVELOPER_GUIDE.md          # This file
```

### What lives outside `eye2voice-ui/`

```
capstone/
├── eye2voice-ui/               # The React app (described above)
└── design-ref/                 # Design specs and reference screenshots
    ├── Home Page.md            # Home page wireframe specs
    ├── responses.md            # Response page design system
    ├── Consent Screen Fixes.md # Consent screen design notes
    ├── Design Issues.md        # Known design issues
    ├── Response Page Design Adjustments.md
    ├── Welcome.md              # Welcome screen specs
    ├── Eye2Voice 2.png         # Wireframe screenshots
    ├── Eye2Voice 3.png
    ├── Eye2Voice 4.png
    ├── IMG_6531.png
    ├── Screenshot *.png         # Implementation reference screenshots
    └── working_home_page.png
```

---

## Screen Flow & Routing

```
WelcomeScreen (/)
    │ [swipe up]
    ▼
MenuScreen (/menu)
    │
    ├──→ Demo (/demo)                    [placeholder — not yet built]
    │
    └──→ ConsentScreen (/consent)
            │ [all 5 boxes checked]
            ▼
         HomePage (/home)
            │ [click any quadrant]
            ▼
         ResponsePage (/response/:id)
            │ [click non-leaf node]
            ▼
         ResponsePage (/response/:id)    [repeats until leaf node reached]
            │ [click leaf node]
            ▼
         [stays on page — final selection]
```

Routes are defined in `src/App.js`:

```jsx
<Route path="/" element={<WelcomeScreen />} />
<Route path="/menu" element={<MenuScreen />} />
<Route path="/demo" element={<div>...</div>} />
<Route path="/consent" element={<ConsentScreen />} />
<Route path="/home" element={<HomePage />} />
<Route path="/response/:id" element={<ResponsePageWrapper />} />
```

### ResponsePageWrapper

`ResponsePage` uses `useState` internally. When navigating from `/response/4` to `/response/17`, React Router reuses the same component instance, so state (like which block is selected) would persist incorrectly. The `ResponsePageWrapper` in `App.js` solves this by passing `key={id}`, which forces React to unmount and remount the component when the `:id` param changes:

```jsx
function ResponsePageWrapper() {
  const { id } = useParams();
  return <ResponsePage key={id} />;
}
```

---

## Component Breakdown

### WelcomeScreen (`/`)

- Displays the Eye2Voice logo (`/public/eye2voice_logo.png`) and tagline "LOOK. DECIDE. SPEAK."
- Detects vertical swipe gestures via `onTouchStart`, `onTouchMove`, `onTouchEnd` handlers
- Requires a minimum 50px upward swipe to trigger navigation to `/menu`
- Has a CSS bouncing arrow animation hinting the user to swipe up

### MenuScreen (`/menu`)

- Two pill-shaped buttons: "Demo of Eye2Voice" and "Consent & Talk"
- Simple navigation — clicking each button calls `navigate('/demo')` or `navigate('/consent')`

### ConsentScreen (`/consent`)

- Displays 5 consent items as toggleable checkboxes (SVG circles)
- 4 session consents + 1 partner consent
- Checkboxes toggle between gray (`#D9D9D9`) and green (`#4CAF50`)
- When all 5 are checked, auto-navigates to `/home` after a 600ms delay

### HomePage (`/home`)

- Reads the 4 root nodes from `responseData.js` via `getRootNodes()`
- Displays them in the quadrant layout: top=Respond, left=Need, right=Ask, bottom=Pain
- On click: selected block turns green, others dim, then navigates to `/response/:id` after 500ms

### ResponsePage (`/response/:id`)

- Reads URL param `:id` to determine which parent node to show children for
- Uses `getChildren(parentId)` to get the 4 child nodes (sorted by position)
- Shows the parent's label as the page title
- **Non-leaf click:** Shows selection feedback, then navigates to `/response/:childId` after 500ms
- **Leaf click:** Shows selection feedback and stays (this is the final selection)

### Shared Layout (QuadrantPage.css)

Both `HomePage` and `ResponsePage` use the same CSS classes for an identical look:

```
┌─────────────────────────────┐
│          [TITLE]            │  ← .quad-title
├─────────────────────────────┤
│                             │
│        TOP BLOCK            │  ← .quad-top (flex: 1)
│                             │
├──────────────┬──────────────┤
│              │              │
│  LEFT BLOCK  │ RIGHT BLOCK  │  ← .quad-middle (flex: 2)
│              │              │    ├── .quad-left
│              │              │    └── .quad-right
├──────────────┴──────────────┤
│                             │
│       BOTTOM BLOCK          │  ← .quad-bottom (flex: 1)
│                             │
└─────────────────────────────┘
```

- Background: teal/blue gradient
- Card: white, 370px wide, `border-radius: 10px`
- Divider lines: 3px solid blue (`#2969AC`)
- Proportions are fixed (1:2:1) — blocks never resize based on content

---

## Response Tree Data Model

### Source of Truth: `data/responses.csv`

The CSV defines the complete response tree:

```csv
id,parent_id,label,is_leaf,position
1,,Respond,false,top
2,,Need,false,left
...
```

| Column | Description |
|--------|-------------|
| `id` | Unique integer identifier |
| `parent_id` | ID of parent node (empty for root nodes) |
| `label` | Display text |
| `is_leaf` | `true` if this is a terminal/final node, `false` if it has children |
| `position` | Which quadrant this node appears in: `top`, `left`, `right`, `bottom` |

### Runtime Data: `src/responseData.js`

The CSV is **not parsed at runtime**. Instead, `responseData.js` contains the same data as a hardcoded JavaScript array. This avoids needing CSV parsing libraries in a Create React App environment.

Each node object:

```js
{ id: 17, parentId: 4, label: 'Physical\nPain', isLeaf: false, position: 'top' }
```

- Labels may contain `\n` for display line breaks (e.g., "Physical\nPain" renders as two lines)
- The `position` field determines which quadrant the node occupies

### Exported Helper Functions

```js
getRootNodes()        // Returns the 4 root nodes sorted by position (top, left, right, bottom)
getChildren(parentId) // Returns all children of a node, sorted by position
getNode(id)           // Returns a single node by ID (O(1) lookup via Map)
```

### Tree Structure Overview

```
Root Level (HomePage)
├── Respond (top)
│   ├── I Understand (top, leaf)
│   ├── No (left, leaf)
│   ├── Yes (right, leaf)
│   └── Repeat That (bottom, leaf)
├── Need (left)
│   ├── Water/Food (top) → Water, Snack, Food, Nothing Right Now (all leaf)
│   ├── Bathroom (left, leaf)
│   ├── Rest (right, leaf)
│   └── Help (bottom, leaf)
├── Ask (right)
│   ├── Call Someone (top) → Family, Friend, Doctor, Nurse (all leaf)
│   ├── Get Nurse/Doctor (left, leaf)
│   ├── Get My Phone (right, leaf)
│   └── Turn on TV (bottom, leaf)
└── Pain (bottom)
    ├── Physical Pain (top) → Head/Arm/Chest/Back → Mild/Moderate/Severe/Unbearable
    ├── Upset (left) → Scared/Frustrated/Sad/Confused (all leaf)
    ├── Not Well (right) → Nausea/Tired/Dizzy/Sore → Mild/Moderate/Severe/Unbearable
    └── Uncomfortable (bottom) → Sensory/Practical/Physical/Medical → [deeper levels]
```

The deepest paths are 4 levels (e.g., Pain → Physical Pain → Head → Severe). Every non-leaf node has exactly 4 children.

### Keeping Data in Sync

If you edit `data/responses.csv`, you must **manually update** `src/responseData.js` to match. The CSV is the design reference; the JS file is what the app actually uses.

---

## Design System

### Colors

| Usage | Color | Hex |
|-------|-------|-----|
| Page background gradient | White → Teal → Blue | `linear-gradient(180deg, #FFF 0%, #01A29C 63.94%, #1685A4 99.99%, #F1F4F6 100%)` |
| Quadrant card | White | `#FFF` |
| Card divider lines | Blue | `#2969AC` / `#2A6AAC` |
| Default label text | Black | `#000` |
| Selected block background | Green | `#59AF54` |
| Selected block text | White | `#FFF` |
| Dimmed block background | Dark teal | `#435C63` |
| Dimmed block text | Black | `#000` |
| App background (other screens) | Dark navy | `#0E1825` |
| Welcome screen background | Light gray | `#F7F7F7` |
| Menu button accent | Light cyan | `#ADE5FD` |
| Consent checkbox (checked) | Green | `#4CAF50` |
| Consent checkbox (unchecked) | Gray | `#D9D9D9` |

### Typography

- **Font Family:** Josefin Sans (loaded from Google Fonts)
- **Weights Used:** 300 (light), 400 (regular), 500 (medium), 600 (semi-bold)
- **Page Title:** 36px, weight 400, black
- **Quadrant Labels:** 24px, weight 400, letter-spacing 2px
- **Welcome Tagline:** 18px, weight 300

### Layout

- **Wireframe target:** 390 x 844px (iPhone viewport)
- **Quadrant card:** 370px wide, up to 690px tall, `border-radius: 10px`
- **Card proportions:** Top 25% | Middle 50% (left + right) | Bottom 25% — enforced with `flex: 1 1 0` / `flex: 2 1 0` and `min-height: 0`
- **Button shapes:** Pill-shaped (border-radius 40-50px) on menu/welcome screens

---

## Design Reference Files

The `design-ref/` directory (at the project root, outside `eye2voice-ui/`) contains Figma export specs and annotated screenshots. These are written in Markdown with embedded images (Obsidian-style `![[image.png]]` syntax).

Key files to consult:
- **`Home Page.md`** — Exact dimensions, gradient values, border specs for the quadrant layout
- **`responses.md`** — How the response tree maps to the UI
- **`Design Issues.md`** — Known problems and fixes needed

---

## State Management

The app uses **no global state** (no Redux, Context API, or similar). All state is local to individual components via React's `useState` hook:

| Component | State | Purpose |
|-----------|-------|---------|
| WelcomeScreen | `touchStartY` | Tracks swipe gesture start position |
| ConsentScreen | `consents` (array of bools) | Tracks which consent boxes are checked |
| HomePage | `selectedId` | Which quadrant was clicked (for selection animation) |
| ResponsePage | `selectedId` | Which quadrant was clicked (for selection animation) |

Navigation between screens is handled by React Router's `useNavigate()` hook. There is no persisted state — refreshing the browser resets everything to the welcome screen.

---

## Testing

Tests use **Jest** (bundled with Create React App) and **React Testing Library**.

```bash
# Run all tests in watch mode
npm test

# Run with coverage
npm test -- --coverage

# Run a specific test file
npm test -- --testPathPattern=ConsentScreen
```

Currently there is only one placeholder test in `App.test.js`. Test files should be placed alongside components with the `.test.js` suffix (e.g., `HomePage.test.js`).

The test setup file (`src/setupTests.js`) imports `@testing-library/jest-dom` which adds matchers like `toBeInTheDocument()`, `toHaveClass()`, etc.

---

## Common Tasks

### Adding a new response node

1. Add the row to `data/responses.csv` with `id`, `parent_id`, `label`, `is_leaf`, and `position`
2. Add the matching object to the `nodes` array in `src/responseData.js`
3. For long labels, add `\n` where you want line breaks (e.g., `'Need\nBlanket'`)
4. Ensure the parent node has `isLeaf: false`
5. Every non-leaf node must have exactly 4 children (one for each position)

### Adding a new screen/route

1. Create `src/NewScreen.js` and `src/NewScreen.css`
2. Import the component in `src/App.js`
3. Add a `<Route path="/new-path" element={<NewScreen />} />` inside the `<Routes>` block
4. Navigate to it from other screens using `useNavigate()`:
   ```js
   const navigate = useNavigate();
   navigate('/new-path');
   ```

### Modifying the quadrant layout

Both `HomePage` and `ResponsePage` share `src/QuadrantPage.css`. Changes to this file affect both pages. Key classes:
- `.quad-card` — The white card container
- `.quad-top` / `.quad-left` / `.quad-right` / `.quad-bottom` — Individual quadrant blocks
- `.quad-middle` — The row containing left and right blocks
- `.quad-selected` / `.quad-dimmed` — Selection state styles

### Integrating the gaze detection model

The current click handlers in `HomePage.js` and `ResponsePage.js` will eventually be replaced (or augmented) by gaze detection events. The selection logic (`handleClick`/`handleSelect`) is where gaze input should feed in:

```js
// In HomePage.js / ResponsePage.js
const handleSelect = (child) => {
  if (selectedId !== null) return;  // Prevent double-selection
  setSelectedId(child.id);         // Trigger visual feedback
  // ... then navigate or stay based on isLeaf
};
```

The model would call this same function when it determines the user is looking at a specific quadrant.
