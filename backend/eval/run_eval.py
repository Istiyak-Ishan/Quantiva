import asyncio
import sys
import argparse
from typing import List

# Setup for running outside of fastapi context
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

async def main():
    parser = argparse.ArgumentParser(description="Run Quantiva Agent Evaluation")
    parser.add_argument("--doc_id", type=str, required=True, help="Document ID to evaluate against")
    parser.add_argument("--user_id", type=str, required=True, help="Owner ID (trusted operator CLI only)")
    parser.add_argument("--n_questions", type=int, default=5, help="Number of questions to generate")
    args = parser.parse_args()
    
    print(f"Starting Eval Run for Document: {args.doc_id}")
    
    from app.db.database import AsyncSessionLocal
    from app.services.eval.eval_dataset import eval_dataset_generator
    from app.services.eval.eval_runner import eval_runner
    
    async with AsyncSessionLocal() as db:
        print(f"1. Generating Golden Dataset (n={args.n_questions})...")
        dataset = await eval_dataset_generator.generate_qa_pairs(args.doc_id, db, n=args.n_questions, user_id=args.user_id)
        print(f"   Generated {len(dataset)} questions.")
        
        if not dataset:
            print("   Failed to generate dataset. Aborting.")
            return
            
        print("2. Running Agent and LLM-as-Judge...")
        report = await eval_runner.run(dataset, [args.doc_id], db, user_id=args.user_id)
        
        print("\n" + "="*50)
        print("EVALUATION REPORT")
        print("="*50)
        print(f"Avg Faithfulness:  {report.avg_faithfulness:.2f} {'✅' if report.avg_faithfulness >= 0.8 else '❌'}")
        print(f"Avg Relevance:     {report.avg_relevance:.2f} {'✅' if report.avg_relevance >= 0.8 else '❌'}")
        print(f"Avg Completeness:  {report.avg_completeness:.2f} {'✅' if report.avg_completeness >= 0.8 else '❌'}")
        
        print("\nFailed Questions (< 0.8 Faithfulness):")
        failed = False
        for i, q in enumerate(report.per_question_scores):
            if q["faithfulness"] < 0.8:
                failed = True
                print(f"\nQ{i+1}: {q['question']}")
                print(f"Score: {q['faithfulness']:.2f}")
                print(f"Reasoning: {q['reasoning']}")
                print(f"Agent Answer: {q['agent_answer']}")
                
        if not failed:
            print("None! Perfect run. 🎉")

if __name__ == "__main__":
    asyncio.run(main())
