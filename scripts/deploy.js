// scripts/deploy.js
// Deploy SupplyEventLogger to localhost (dev) or PureChain (final)

const hre = require("hardhat");

async function main() {
  console.log("Deploying SupplyEventLogger...");
  console.log("Network:", hre.network.name);

  const [deployer] = await hre.ethers.getSigners();
  console.log("Deployer:", deployer.address);

  const Logger = await hre.ethers.getContractFactory("SupplyEventLogger");
  const logger = await Logger.deploy();
  await logger.waitForDeployment();

  const address = await logger.getAddress();
  console.log("SupplyEventLogger deployed to:", address);
  console.log("\nAdd this to your .env file:");
  console.log(`SUPPLY_EVENT_LOGGER_ADDRESS=${address}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
