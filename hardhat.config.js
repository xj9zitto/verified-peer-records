require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

const purechainKey = process.env.PURECHAIN_PRIVATE_KEY;
const validKey = purechainKey && purechainKey.length === 66;

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.24",
    settings: {
      viaIR: true,
      optimizer: {
        enabled: true,
        runs: 200,
      },
    },
  },
  networks: {
    localhost: {
      url: "http://127.0.0.1:8545",
      chainId: 31337,
    },
    purechain: {
      // Explicit URL — does not depend on env loading order
      url: "https://purechainnode.com",
      chainId: 900520900520,
      accounts: validKey ? [purechainKey] : [],
      gasPrice: 0,
      gas: 8000000,
      timeout: 60000,    // 60s — remote network may be slower than localhost
    },
  },
  paths: {
    sources: "./contracts",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts",
  },
};