import type { SiteConfig } from '@mcptoolshop/site-theme';

export const config: SiteConfig = {
  title: 'ai-eyes',
  description:
    'Grounded visual evaluator MCP server. Honest image judgment via SigLIP2 on a pinned model revision — it measures, it does not narrate.',
  logoBadge: 'AE',
  brandName: 'ai-eyes',
  repoUrl: 'https://github.com/mcp-tool-shop-org/ai-eyes-mcp',
  footerText:
    'MIT Licensed — built by <a href="https://github.com/mcp-tool-shop-org" style="color:var(--color-muted);text-decoration:underline">mcp-tool-shop-org</a>',

  hero: {
    badge: 'v1.2.0 · 8 MCP tools · 207 tests',
    headline: 'Your model can look at the image.',
    headlineAccent: 'This one can be wrong about it.',
    description:
      'Generative vision models hallucinate confident answers because they complete narratives. ai-eyes measures one image-text pair with SigLIP2 and returns a calibrated score — on weights it names, with a flag when it only saw part of your query, and a refusal when the number would not mean anything.',
    primaryCta: { href: '#usage', label: 'Get started' },
    secondaryCta: { href: 'handbook/', label: 'Read the Handbook' },
    previews: [
      { label: 'Ask', code: 'image_contains("sprite.png", "a knight with a sword")' },
      { label: 'Get', code: '{ "present": true, "score": 0.6847,\n  "truncated": false,\n  "revision": "e8e4872..." }' },
      { label: 'Rank', code: 'image_rank(reference="canon.png",\n           candidates=[...],\n           baselines=[["a.png","b.png"]])' },
    ],
  },

  sections: [
    {
      kind: 'features',
      id: 'features',
      title: 'Measurement, not narration',
      subtitle: 'One claim, defended in the code rather than the copy.',
      features: [
        {
          title: 'It names its weights',
          desc: 'The model revision is pinned to a commit SHA and reported in every payload that contains a number. A floating branch means two installs can disagree and never say so — this one refuses branch names outright.',
        },
        {
          title: 'It tells you what it did not read',
          desc: 'The text encoder holds 64 tokens. A longer query is scored on its prefix — so the payload carries truncated: true rather than presenting a partial measurement as a whole one.',
        },
        {
          title: 'It abstains',
          desc: 'Cosine similarity between two different characters in one art style measured 0.698–0.836. There is no universal cutoff, so image_compare and image_rank take contrast pairs from you and return incomplete rather than inventing a threshold.',
        },
        {
          title: 'It will not contradict itself',
          desc: 'Verdicts are decided at full precision and rounded only for display, never far enough to disagree with the number printed beside them. A measured non-zero never prints as zero.',
        },
        {
          title: 'Local, and quiet about it',
          desc: 'All inference runs on your machine. The one network call is the first-run model download; point AI_EYES_MODEL_DIR at a pre-populated cache and there are none at all.',
        },
        {
          title: 'Proves itself on install',
          desc: 'eyes_selftest runs known orderings on bundled reference images and reports the revision it used — so a broken cache or a swapped model surfaces immediately, not as quietly wrong scores.',
        },
      ],
    },
    {
      kind: 'code-cards',
      id: 'usage',
      title: 'Usage',
      cards: [
        {
          title: 'Install',
          code: 'git clone https://github.com/mcp-tool-shop-org/ai-eyes-mcp\ncd ai-eyes-mcp\npip install -e .',
        },
        {
          title: 'Register with Claude Code',
          code: '{\n  "mcpServers": {\n    "ai-eyes": {\n      "command": "ai-eyes-mcp",\n      "env": { "AI_EYES_MODEL_DIR": "/path/to/cache" }\n    }\n  }\n}',
        },
        {
          title: 'Verify a claim about pixels',
          code: 'image_verify(\n  image_path="sprite.png",\n  target="a knight with a sword",\n  alternatives=["a goblin cook", "a bard"],\n)\n# → decision + margin + a confidence band\n#   that describes the gap it measured',
        },
        {
          title: 'Find the closest match, or none',
          code: 'image_rank(\n  reference="canon/knight.png",\n  candidates=generated_frames,\n  baselines=[["knight.png", "cook.png"]],\n)\n# matches: [] when nothing clears the floor\n# you supplied. No forced top-k.',
        },
      ],
    },
  ],
};
