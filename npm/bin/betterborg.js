#!/usr/bin/env node

"use strict";

const metadata = require("../package.json");
const { main, reportFailure } = require("../lib/launcher");

main(process.argv.slice(2), { version: metadata.version }).catch(reportFailure);
