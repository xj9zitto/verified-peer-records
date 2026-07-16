// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * SupplyEventLogger.sol
 * Records Type-A (asset state) and Type-B (routing condition)
 * supply events from Layer 1 onto the PureChain ledger.
 * Gas-free under PureChain PoA² consensus.
 */
contract SupplyEventLogger {

    // ── Event definitions ─────────────────────────────────────

    event AssetStateLogged(
        bytes32 indexed eventId,
        string  assetId,
        uint256 timestamp,
        string  hubLocation,
        uint256 socBasisPoints,       // SOC × 100 (e.g. 1820 = 18.20%)
        int256  temperatureCelsius,   // temp × 100 (e.g. 3100 = 31.00°C)
        uint256 degradationBasisPoints, // degradation × 100
        bytes32 sensorHash
    );

    event RoutingConditionLogged(
        bytes32 indexed eventId,
        string  pathId,
        string  hopOrigin,
        string  hopDest,
        uint256 transitHours,
        int256  ambTempMinCelsius,
        int256  ambTempMaxCelsius,
        bool    tempControlled,
        string  carrierId,
        uint256 carrierVarianceBasisPoints // variance × 100
    );

    event DecisionLogged(
        bytes32 indexed decisionId,
        string  classification,  // "valid" | "invalid" | "uncertain"
        string  decisionType,    // "reorder" | "reroute" | "carrier"
        uint256 tracerate,       // trace rate × 10000
        uint256 timestamp
    );

    // ── Storage ───────────────────────────────────────────────

    address public owner;
    uint256 public assetEventCount;
    uint256 public routingEventCount;
    uint256 public decisionCount;

    // eventId → block number for auditability
    mapping(bytes32 => uint256) public eventBlocks;

    // ── Access control ────────────────────────────────────────

    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorised");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    // ── Type-A: Asset state event ─────────────────────────────

    function logAssetState(
        string  calldata assetId,
        string  calldata hubLocation,
        uint256 socBasisPoints,
        int256  temperatureCelsius,
        uint256 degradationBasisPoints,
        bytes32 sensorHash
    ) external onlyOwner returns (bytes32 eventId) {
        eventId = keccak256(abi.encodePacked(
            "ASSET", assetId, block.timestamp, assetEventCount
        ));
        eventBlocks[eventId] = block.number;
        assetEventCount++;

        emit AssetStateLogged(
            eventId,
            assetId,
            block.timestamp,
            hubLocation,
            socBasisPoints,
            temperatureCelsius,
            degradationBasisPoints,
            sensorHash
        );
    }

    // ── Type-B: Routing condition event ───────────────────────

    function logRoutingCondition(
        string  calldata pathId,
        string  calldata hopOrigin,
        string  calldata hopDest,
        uint256 transitHours,
        int256  ambTempMin,
        int256  ambTempMax,
        bool    tempControlled,
        string  calldata carrierId,
        uint256 carrierVarianceBasisPoints
    ) external onlyOwner returns (bytes32 eventId) {
        eventId = keccak256(abi.encodePacked(
            "ROUTING", pathId, hopOrigin, hopDest,
            block.timestamp, routingEventCount
        ));
        eventBlocks[eventId] = block.number;
        routingEventCount++;

        emit RoutingConditionLogged(
            eventId,
            pathId,
            hopOrigin,
            hopDest,
            transitHours,
            ambTempMin,
            ambTempMax,
            tempControlled,
            carrierId,
            carrierVarianceBasisPoints
        );
    }

    // ── Decision audit log ────────────────────────────────────

    function logDecision(
        string  calldata classification,
        string  calldata decisionType,
        uint256 traceRate
    ) external onlyOwner returns (bytes32 decisionId) {
        decisionId = keccak256(abi.encodePacked(
            "DECISION", block.timestamp, decisionCount
        ));
        eventBlocks[decisionId] = block.number;
        decisionCount++;

        emit DecisionLogged(
            decisionId,
            classification,
            decisionType,
            traceRate,
            block.timestamp
        );
    }

    // ── Verification helper ───────────────────────────────────

    function verifyEvent(bytes32 eventId) external view returns (
        bool exists, uint256 blockNumber
    ) {
        blockNumber = eventBlocks[eventId];
        exists = blockNumber > 0;
    }
}
