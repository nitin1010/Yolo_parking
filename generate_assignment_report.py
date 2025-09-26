"""
Generate final parking assignment report from JSON data
Creates comprehensive reports with license plate, timestamp, and parking slot assignments
"""

import json
import os
from datetime import datetime
from typing import List, Dict


def generate_assignment_report(assignments_file: str = "parking_assignments.json",
                               output_file: str = "parking_report.json") -> None:
    """Generate a comprehensive parking assignment report"""

    print("=== Parking Assignment Report Generator ===")

    # Check if assignments file exists
    if not os.path.exists(assignments_file):
        print(f"❌ No assignments file found: {assignments_file}")
        print("   Please run the parking system first to generate assignments.")
        return

    try:
        with open(assignments_file, 'r') as f:
            assignments = json.load(f)
    except Exception as e:
        print(f"❌ Error loading assignments: {e}")
        return

    if not assignments:
        print("❌ No assignments found in the file")
        return

    # Generate comprehensive report
    report = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "source_file": assignments_file,
            "report_version": "1.0"
        },
        "summary_statistics": {
            "total_assignments": len(assignments),
            "unique_license_plates": len(set(a["license_plate"] for a in assignments)),
            "parking_slots_used": list(set(a["parking_slot"] for a in assignments)),
            "active_assignments": len([a for a in assignments if a.get("status") == "assigned"]),
            "first_assignment_time": assignments[0]["timestamp"] if assignments else None,
            "last_assignment_time": assignments[-1]["timestamp"] if assignments else None
        },
        "detailed_assignments": assignments,
        "slot_utilization": {}
    }

    # Calculate slot utilization statistics
    slot_counts = {}
    for assignment in assignments:
        slot = assignment["parking_slot"]
        slot_counts[slot] = slot_counts.get(slot, 0) + 1

    report["slot_utilization"] = {
        "assignments_per_slot": slot_counts,
        "most_used_slot": max(slot_counts, key=slot_counts.get) if slot_counts else None,
        "least_used_slot": min(slot_counts, key=slot_counts.get) if slot_counts else None
    }

    # Save comprehensive report
    try:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ Comprehensive report generated: {output_file}")

    except Exception as e:
        print(f"❌ Error saving report: {e}")
        return

    # Print summary to console
    print(f"\n📊 PARKING ASSIGNMENT SUMMARY")
    print(f"{'=' * 50}")
    print(f"📈 Total Assignments: {report['summary_statistics']['total_assignments']}")
    print(f"🚗 Unique License Plates: {report['summary_statistics']['unique_license_plates']}")
    print(f"🅿️  Parking Slots Used: {len(report['summary_statistics']['parking_slots_used'])}")
    print(f"✅ Active Assignments: {report['summary_statistics']['active_assignments']}")

    if report['summary_statistics']['first_assignment_time']:
        print(
            f"⏰ Time Period: {report['summary_statistics']['first_assignment_time'][:19]} to {report['summary_statistics']['last_assignment_time'][:19]}")

    # Show slot utilization
    print(f"\n🎯 SLOT UTILIZATION")
    print(f"{'=' * 30}")
    for slot, count in sorted(slot_counts.items()):
        print(f"   {slot}: {count} assignment{'s' if count > 1 else ''}")

    if report["slot_utilization"]["most_used_slot"]:
        most_used = report["slot_utilization"]["most_used_slot"]
        print(f"🏆 Most Used Slot: {most_used} ({slot_counts[most_used]} assignments)")

    # Show recent assignments
    print(f"\n📋 RECENT ASSIGNMENTS")
    print(f"{'=' * 40}")
    recent_assignments = assignments[-10:]  # Last 10 assignments

    for assignment in recent_assignments:
        plate = assignment["license_plate"]
        slot = assignment["parking_slot"]
        timestamp = assignment["timestamp"][:19]  # Remove microseconds
        assignment_id = assignment.get("assignment_id", "N/A")
        status = assignment.get("status", "unknown")

        print(f"   [{assignment_id:2}] {plate} → {slot} | {timestamp} | {status}")

    print(f"\n💾 Full report saved to: {output_file}")
    print(f"🔍 Contains {len(assignments)} detailed assignment records")


def generate_simple_csv_report(assignments_file: str = "parking_assignments.json",
                               csv_output: str = "parking_assignments.csv") -> None:
    """Generate a simple CSV report for Excel/spreadsheet use"""

    if not os.path.exists(assignments_file):
        return

    try:
        with open(assignments_file, 'r') as f:
            assignments = json.load(f)
    except:
        return

    if not assignments:
        return

    # Generate CSV content
    csv_content = "Assignment_ID,License_Plate,Parking_Slot,Timestamp,Status\n"

    for assignment in assignments:
        assignment_id = assignment.get("assignment_id", "")
        license_plate = assignment.get("license_plate", "")
        parking_slot = assignment.get("parking_slot", "")
        timestamp = assignment.get("timestamp", "")
        status = assignment.get("status", "")

        csv_content += f"{assignment_id},{license_plate},{parking_slot},{timestamp},{status}\n"

    try:
        with open(csv_output, 'w') as f:
            f.write(csv_content)
        print(f"📊 CSV report generated: {csv_output}")
    except Exception as e:
        print(f"❌ Error generating CSV: {e}")


if __name__ == "__main__":
    # Generate both JSON and CSV reports
    generate_assignment_report()
    generate_simple_csv_report()

    print(f"\n🎉 Report generation complete!")
    print(f"📁 Check the following files:")
    print(f"   • parking_report.json (comprehensive JSON report)")
    print(f"   • parking_assignments.csv (simple CSV for Excel)")