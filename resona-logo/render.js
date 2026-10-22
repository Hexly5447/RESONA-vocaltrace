// Usage: node render.js input.svg output.png width [backgroundColor]
const { Resvg } = require('@resvg/resvg-js');
const fs = require('fs');
const [, , input, output, width = '1024', bg] = process.argv;
const svg = fs.readFileSync(input);
const opts = { fitTo: { mode: 'width', value: parseInt(width, 10) } };
if (bg && bg !== 'none') opts.background = bg;
const r = new Resvg(svg, opts);
fs.writeFileSync(output, r.render().asPng());
console.log('wrote', output);
