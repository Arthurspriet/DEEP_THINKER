"""
Claims CLI for manual extraction and inspection.

Run with: python -m deepthinker.claims [OPTIONS]

Commands:
    --mission-id ID           Extract claims for a specific mission
    --from-file PATH          Extract from a file
    --mode MODE               Extraction mode (regex, hf, llm-json)
    --batch-summaries         Process all summaries in kb/long_memory/summaries.json
    --list                    List all missions with claims
    --stats                   Show claim store statistics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def extract_from_mission(
    mission_id: str,
    mode: Optional[str] = None,
    from_file: Optional[str] = None,
) -> int:
    """Extract claims for a mission."""
    from .store import extract_and_store_claims, get_claim_store
    
    # Get text to extract from
    if from_file:
        text = Path(from_file).read_text(encoding="utf-8")
        source_type = "file"
        source_ref = from_file
    else:
        # Try to load from mission state
        state_path = Path("kb/missions") / mission_id / "state.json"
        if not state_path.exists():
            print(f"Error: Mission state not found at {state_path}")
            print("Use --from-file to specify a file to extract from")
            return 1
        
        with open(state_path, "r") as f:
            state = json.load(f)
        
        # Extract from final artifacts or last phase output
        text_parts = []
        
        # Try final_artifacts
        final_artifacts = state.get("final_artifacts", {})
        for key, value in final_artifacts.items():
            if isinstance(value, str) and len(value) > 50:
                text_parts.append(f"# {key}\n{value}")
        
        if not text_parts:
            print(f"No extractable text found in mission state")
            return 1
        
        text = "\n\n".join(text_parts)
        source_type = "final_answer"
        source_ref = str(state_path)
    
    print(f"Extracting claims from {len(text)} characters...")
    print(f"Mode: {mode or 'default (from config)'}")
    
    result = extract_and_store_claims(
        text=text,
        mission_id=mission_id,
        source_type=source_type,
        source_ref=source_ref,
        mode=mode,
    )
    
    print(f"\nExtraction complete:")
    print(f"  Mode used: {result.extractor_mode}")
    print(f"  Claims extracted: {result.claim_count}")
    print(f"  Time: {result.extraction_time_ms:.1f}ms")
    
    if result.error:
        print(f"  Error: {result.error}")
    
    store = get_claim_store()
    claims_path = store._get_claims_path(mission_id)
    print(f"  Saved to: {claims_path}")
    
    # Show sample claims
    if result.claims:
        print(f"\nSample claims (first 3):")
        for claim in result.claims[:3]:
            text_preview = claim.text[:80] + "..." if len(claim.text) > 80 else claim.text
            print(f"  - [{claim.claim_type}] {text_preview}")
    
    return 0


def batch_process_summaries(mode: Optional[str] = None) -> int:
    """Process all summaries in kb/long_memory/summaries.json."""
    from .store import extract_and_store_claims
    
    summaries_path = Path("kb/long_memory/summaries.json")
    if not summaries_path.exists():
        print(f"Error: Summaries file not found at {summaries_path}")
        return 1
    
    with open(summaries_path, "r") as f:
        data = json.load(f)
    
    summaries = data.get("summaries", [])
    if not summaries:
        print("No summaries found")
        return 0
    
    print(f"Processing {len(summaries)} summaries...")
    
    total_claims = 0
    processed = 0
    
    for summary in summaries:
        mission_id = summary.get("mission_id")
        if not mission_id:
            continue
        
        # Build text from summary fields
        text_parts = [
            summary.get("objective", ""),
        ]
        
        key_insights = summary.get("key_insights", [])
        if key_insights:
            text_parts.extend(key_insights)
        
        text = "\n".join(filter(None, text_parts))
        
        if len(text) < 50:
            continue
        
        result = extract_and_store_claims(
            text=text,
            mission_id=mission_id,
            source_type="summary",
            source_ref="kb/long_memory/summaries.json",
            mode=mode,
        )
        
        total_claims += result.claim_count
        processed += 1
        print(f"  {mission_id[:8]}...: {result.claim_count} claims ({result.extractor_mode})")
    
    print(f"\nProcessed {processed} summaries, extracted {total_claims} claims")
    return 0


def list_missions() -> int:
    """List all missions with claims."""
    from .store import get_claim_store
    
    store = get_claim_store()
    missions = store.list_missions()
    
    if not missions:
        print("No claims found")
        return 0
    
    print(f"Missions with claims ({len(missions)}):\n")
    
    for mission_id in missions:
        claim_count = store.get_claim_count(mission_id)
        runs = store.read_run_metadata(mission_id)
        run_count = len(runs)
        
        modes = set(r.get("extractor_mode", "unknown") for r in runs)
        modes_str = ", ".join(sorted(modes))
        
        print(f"  {mission_id}: {claim_count} claims, {run_count} runs ({modes_str})")
    
    return 0


def show_stats() -> int:
    """Show claim store statistics."""
    from .store import get_claim_store
    
    store = get_claim_store()
    stats = store.get_stats()
    
    print("Claim Store Statistics")
    print("=" * 40)
    print(f"  Base directory: {stats['base_dir']}")
    print(f"  Missions: {stats['mission_count']}")
    print(f"  Total claims: {stats['total_claims']}")
    print(f"  Total runs: {stats['total_runs']}")
    
    return 0


def show_claims(mission_id: str) -> int:
    """Show claims for a mission."""
    from .store import get_claim_store
    
    store = get_claim_store()
    claims = store.read_claims(mission_id)
    
    if not claims:
        print(f"No claims found for mission {mission_id}")
        return 0
    
    print(f"Claims for mission {mission_id} ({len(claims)}):\n")
    
    for i, claim in enumerate(claims, 1):
        print(f"{i}. [{claim.get('claim_type', 'unknown')}]")
        print(f"   {claim.get('text', 'No text')}")
        if claim.get('confidence') is not None:
            print(f"   Confidence: {claim.get('confidence'):.2f}")
        if claim.get('entities'):
            print(f"   Entities: {', '.join(claim.get('entities', []))}")
        print()
    
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Claims extraction and management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--mission-id",
        help="Extract claims for a specific mission",
    )
    parser.add_argument(
        "--from-file",
        help="Extract from a specific file",
    )
    parser.add_argument(
        "--mode",
        choices=["regex", "hf", "llm-json"],
        help="Extraction mode (default: from config)",
    )
    parser.add_argument(
        "--batch-summaries",
        action="store_true",
        help="Process all summaries in kb/long_memory/summaries.json",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all missions with claims",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show claim store statistics",
    )
    parser.add_argument(
        "--show",
        metavar="MISSION_ID",
        help="Show claims for a mission",
    )
    
    args = parser.parse_args()
    
    # Handle commands
    if args.list:
        return list_missions()
    
    if args.stats:
        return show_stats()
    
    if args.show:
        return show_claims(args.show)
    
    if args.batch_summaries:
        return batch_process_summaries(args.mode)
    
    if args.mission_id:
        return extract_from_mission(
            mission_id=args.mission_id,
            mode=args.mode,
            from_file=args.from_file,
        )
    
    # No command specified
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

