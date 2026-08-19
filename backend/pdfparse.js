const fs = require('node:fs/promises');
const path = require('node:path');
const pdf = require('pdf-parse');
const { fromPath } = require('pdf2pic');

async function convertPdfToMarkdown() {
  const args = process.argv.slice(2);
  const inputPdfPath = args[0];
  const outputMdPath = args[1];

  if (!inputPdfPath || !outputMdPath) {
    console.error('Usage: node pdfparse.js <input-pdf> <output-md>');
    process.exit(1);
  }

  try {
    console.log(`Extracting text from "${inputPdfPath}"...`);

    // 1. Read & Extract Text
    const dataBuffer = await fs.readFile(inputPdfPath);
    const pdfData = await pdf(dataBuffer);

    // 2. Prepare output image directory
    const outputDir = path.dirname(path.resolve(outputMdPath));
    const imgFolder = path.join(outputDir, 'images');
    await fs.mkdir(imgFolder, { recursive: true });

    // 3. Configure PDF2PIC Options
    console.log(`Rendering ${pdfData.numpages} page(s) to PNG in "${imgFolder}"...`);

    const options = {
      density: 150,
      saveFilename: 'page',
      savePath: imgFolder,
      format: 'png',
      width: 1200,
      height: 1600
    };

    const storeAsImage = fromPath(inputPdfPath, options);

    // 4. Convert ALL pages cleanly (-1 converts all pages)
    const pageResults = await storeAsImage.bulk(-1, { responseType: 'image' });

    let imageMarkdownLinks = '\n\n## Page Images\n\n';

    if (Array.isArray(pageResults)) {
      pageResults.forEach((page, index) => {
        // Use actual filename returned by pdf2pic or fallback
        const filename = page.name || `page.${index + 1}.png`;
        imageMarkdownLinks += `![Page ${index + 1}](./images/${filename})\n\n`;
      });
    }

    // 5. Save the Markdown
    const finalMarkdown = `# Extracted Document\n\n**Total Pages:** ${pdfData.numpages}\n\n---\n\n${pdfData.text}${imageMarkdownLinks}`;

    await fs.writeFile(outputMdPath, finalMarkdown, 'utf-8');
    console.log(`\nSuccess! Saved "${outputMdPath}" and generated ${pageResults.length} image(s) in "${imgFolder}".`);

  } catch (error) {
    console.error('Conversion failed:', error.message);
  }
}

convertPdfToMarkdown();