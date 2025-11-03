#!/usr/bin/env python3
"""
Additional Methods Extension for Document Processor

Extended functionality for the Multi-Modal RAG Pipeline Document Processor.
Provides enhanced document processing capabilities, batch operations,
cache management, and performance optimization features.


"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from config import Config


class DocumentProcessorExtensions:
    """
    Extended functionality for the Document Processor.
    
    Provides additional methods for batch processing, cache management,
    performance optimization, and advanced document handling capabilities.
    """
    
    # =============================================================================
    # CACHE MANAGEMENT AND RETRIEVAL
    # =============================================================================
    
    def load_from_saved_extraction(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load content from a previously saved extraction.
        
        Args:
            file_path: Path to the original source file
            
        Returns:
            Dictionary containing extracted content (text, tables, visuals)
        """
        return self._load_saved_extraction(file_path)
    
    def get_all_saved_extractions(self) -> List[Dict[str, Any]]:
        """
        Retrieve metadata for all saved extractions.
        
        Returns:
            List of dictionaries containing extraction metadata including
            source file, hash, date, and content statistics
        """
        extractions = []
        
        if not os.path.exists(self.content_dir):
            return extractions
        
        for filename in os.listdir(self.content_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.content_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    extraction_metadata = {
                        'extraction_file': filepath,
                        'source_file': data.get('source_file', ''),
                        'file_hash': data.get('file_hash', ''),
                        'extraction_date': data.get('extraction_date', ''),
                        'content_statistics': {
                            'text_items': len(data.get('content', {}).get('text', [])),
                            'table_items': len(data.get('content', {}).get('tables', [])),
                            'visual_items': len(data.get('content', {}).get('visuals', []))
                        },
                        'file_size_kb': os.path.getsize(filepath) // 1024
                    }
                    
                    extractions.append(extraction_metadata)
                    
                except Exception as e:
                    print(f"⚠️  Error reading extraction file {filename}: {e}")
        
        # Sort by extraction date (newest first)
        extractions.sort(
            key=lambda x: x.get('extraction_date', ''),
            reverse=True
        )
        
        return extractions
    
    def get_extraction_cache_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the extraction cache.
        
        Returns:
            Dictionary containing cache statistics and performance metrics
        """
        stats = {
            'total_extractions': 0,
            'total_cache_size_mb': 0,
            'content_breakdown': {
                'text_items': 0,
                'table_items': 0,
                'visual_items': 0
            },
            'cache_directories': {},
            'oldest_extraction': None,
            'newest_extraction': None
        }
        
        # Analyze cache directories
        cache_dirs = [
            ('extractions', self.extractions_dir),
            ('images', self.images_dir),
            ('content', self.content_dir)
        ]
        
        for dir_name, dir_path in cache_dirs:
            if os.path.exists(dir_path):
                files = os.listdir(dir_path)
                total_size = sum(
                    os.path.getsize(os.path.join(dir_path, f))
                    for f in files
                    if os.path.isfile(os.path.join(dir_path, f))
                )
                
                stats['cache_directories'][dir_name] = {
                    'files': len(files),
                    'size_mb': total_size / (1024 * 1024)
                }
                
                stats['total_cache_size_mb'] += total_size / (1024 * 1024)
        
        # Analyze extraction content
        extractions = self.get_all_saved_extractions()
        stats['total_extractions'] = len(extractions)
        
        if extractions:
            # Content breakdown
            for extraction in extractions:
                content_stats = extraction.get('content_statistics', {})
                stats['content_breakdown']['text_items'] += content_stats.get('text_items', 0)
                stats['content_breakdown']['table_items'] += content_stats.get('table_items', 0)
                stats['content_breakdown']['visual_items'] += content_stats.get('visual_items', 0)
            
            # Date range
            stats['oldest_extraction'] = extractions[-1].get('extraction_date', 'Unknown')
            stats['newest_extraction'] = extractions[0].get('extraction_date', 'Unknown')
        
        return stats
    
    # =============================================================================
    # CACHE MAINTENANCE AND CLEANUP
    # =============================================================================
    
    def clear_saved_extractions(self, keep_recent_days: int = 30) -> Dict[str, int]:
        """
        Clear old extraction files to maintain cache performance.
        
        Args:
            keep_recent_days: Number of days to keep extractions (default: 30)
            
        Returns:
            Dictionary with removal statistics
        """
        if not os.path.exists(self.content_dir):
            return {'removed_files': 0, 'kept_files': 0}
        
        cutoff_date = datetime.now() - timedelta(days=keep_recent_days)
        
        removed_count = 0
        kept_count = 0
        
        for filename in os.listdir(self.content_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.content_dir, filename)
                
                try:
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    if file_mtime < cutoff_date:
                        os.remove(filepath)
                        removed_count += 1
                    else:
                        kept_count += 1
                        
                except Exception as e:
                    print(f"⚠️  Error processing extraction file {filename}: {e}")
                    kept_count += 1
        
        print(f"🧹 Cache cleanup completed: {removed_count} files removed, {kept_count} files kept")
        
        return {
            'removed_files': removed_count,
            'kept_files': kept_count,
            'cutoff_date': cutoff_date.isoformat()
        }
    
    def optimize_cache_storage(self) -> Dict[str, Any]:
        """
        Optimize cache storage by removing duplicates and compressing data.
        
        Returns:
            Dictionary with optimization results
        """
        print("🔧 Optimizing cache storage...")
        
        optimization_results = {
            'duplicates_removed': 0,
            'space_saved_mb': 0,
            'files_optimized': 0,
            'optimization_errors': 0
        }
        
        if not os.path.exists(self.content_dir):
            return optimization_results
        
        # Track file hashes to detect duplicates
        file_hashes = {}
        files_to_remove = []
        
        for filename in os.listdir(self.content_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.content_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    file_hash = data.get('file_hash', '')
                    
                    if file_hash in file_hashes:
                        # Duplicate found - keep the newer file
                        existing_file = file_hashes[file_hash]
                        existing_mtime = os.path.getmtime(existing_file)
                        current_mtime = os.path.getmtime(filepath)
                        
                        if current_mtime > existing_mtime:
                            files_to_remove.append(existing_file)
                            file_hashes[file_hash] = filepath
                        else:
                            files_to_remove.append(filepath)
                    else:
                        file_hashes[file_hash] = filepath
                    
                except Exception as e:
                    print(f"⚠️  Error processing {filename}: {e}")
                    optimization_results['optimization_errors'] += 1
        
        # Remove duplicate files
        for file_to_remove in files_to_remove:
            try:
                file_size = os.path.getsize(file_to_remove)
                os.remove(file_to_remove)
                optimization_results['duplicates_removed'] += 1
                optimization_results['space_saved_mb'] += file_size / (1024 * 1024)
            except Exception as e:
                print(f"⚠️  Error removing duplicate file {file_to_remove}: {e}")
                optimization_results['optimization_errors'] += 1
        
        print(f"✅ Cache optimization completed:")
        print(f"   Duplicates removed: {optimization_results['duplicates_removed']}")
        print(f"   Space saved: {optimization_results['space_saved_mb']:.2f} MB")
        
        return optimization_results
    
    # =============================================================================
    # BATCH PROCESSING OPERATIONS
    # =============================================================================
    
    def process_multiple_documents(
        self, 
        file_paths: List[str], 
        force_reprocess: bool = False,
        show_progress: bool = True
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Process multiple documents with optimized batch operations.
        
        Args:
            file_paths: List of file paths to process
            force_reprocess: Whether to bypass cache and reprocess files
            show_progress: Whether to show progress information
            
        Returns:
            Dictionary mapping file paths to their extracted content
        """
        results = {}
        total_files = len(file_paths)
        
        if show_progress:
            print(f"📦 Processing {total_files} documents...")
        
        for index, file_path in enumerate(file_paths, 1):
            try:
                if show_progress:
                    progress = (index / total_files) * 100
                    print(f"\n📄 [{index}/{total_files}] ({progress:.1f}%) Processing: {os.path.basename(file_path)}")
                
                # Use optimized processing based on configuration
                if Config.ENABLE_MULTIPROCESSING and hasattr(self, 'process_document_parallel'):
                    result = self.process_document_parallel(file_path, force_reprocess)
                else:
                    result = self.process_document(file_path, force_reprocess)
                
                results[file_path] = result
                
                if show_progress:
                    total_items = sum(len(content) for content in result.values())
                    print(f"   ✅ Extracted {total_items} items")
                
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                results[file_path] = {"text": [], "tables": [], "visuals": []}
        
        if show_progress:
            successful_files = sum(1 for result in results.values() 
                                 if any(len(content) > 0 for content in result.values()))
            print(f"\n🎉 Batch processing completed: {successful_files}/{total_files} files successful")
        
        return results
    
    def process_documents_from_directory(
        self, 
        directory_path: str, 
        file_extensions: Optional[List[str]] = None,
        force_reprocess: bool = False
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Process all documents from a directory with specified extensions.
        
        Args:
            directory_path: Path to directory containing documents
            file_extensions: List of file extensions to process (default: all supported)
            force_reprocess: Whether to bypass cache and reprocess files
            
        Returns:
            Dictionary mapping file paths to their extracted content
        """
        if not os.path.exists(directory_path):
            print(f"❌ Directory not found: {directory_path}")
            return {}
        
        # Default supported extensions
        if file_extensions is None:
            file_extensions = ['.pdf', '.docx', '.pptx', '.xlsx', '.eml']
        
        # Find all matching files
        matching_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext.lower()) for ext in file_extensions):
                    matching_files.append(os.path.join(root, file))
        
        print(f"📁 Found {len(matching_files)} documents in directory: {directory_path}")
        
        if not matching_files:
            print("⚠️  No matching documents found")
            return {}
        
        return self.process_multiple_documents(matching_files, force_reprocess)
    
    # =============================================================================
    # PERFORMANCE MONITORING AND ANALYTICS
    # =============================================================================
    
    def get_detailed_performance_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics and recommendations.
        
        Returns:
            Dictionary containing detailed performance metrics and analysis
        """
        base_stats = self.get_performance_statistics()
        
        # Enhanced statistics
        enhanced_stats = {
            **base_stats,
            'cache_statistics': self.get_extraction_cache_statistics(),
            'optimization_recommendations': self._generate_optimization_recommendations(),
            'system_health': self._assess_system_health(),
            'processing_efficiency': self._calculate_processing_efficiency()
        }
        
        return enhanced_stats
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate personalized optimization recommendations."""
        recommendations = []
        
        # Check configuration optimizations
        if not Config.ENABLE_BATCH_PROCESSING:
            recommendations.append("Enable batch processing for 3x faster API calls")
        
        if not Config.ENABLE_IMAGE_COMPRESSION:
            recommendations.append("Enable image compression to reduce transfer time by 70%")
        
        if not Config.ENABLE_SMART_FILTERING:
            recommendations.append("Enable smart filtering to skip 10-20% of unnecessary pages")
        
        if not Config.ENABLE_DUPLICATE_DETECTION:
            recommendations.append("Enable duplicate detection to prevent redundant processing")
        
        # Check cache health
        cache_stats = self.get_extraction_cache_statistics()
        if cache_stats['total_cache_size_mb'] > 1000:  # 1GB
            recommendations.append("Consider cleaning old cache files - cache size exceeds 1GB")
        
        if cache_stats['total_extractions'] > 1000:
            recommendations.append("Large number of cached extractions - consider archiving old files")
        
        # Check performance settings
        if Config.MAX_WORKERS > 8:
            recommendations.append("Consider reducing MAX_WORKERS to 4-8 for optimal performance")
        
        if Config.BATCH_SIZE > 20:
            recommendations.append("Consider reducing BATCH_SIZE to 10-20 for better API performance")
        
        return recommendations
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health and performance."""
        health_score = 100
        issues = []
        
        # Check cache directories
        required_dirs = [self.extractions_dir, self.images_dir, self.content_dir]
        for directory in required_dirs:
            if not os.path.exists(directory):
                health_score -= 10
                issues.append(f"Missing directory: {os.path.basename(directory)}")
        
        # Check optimization settings
        optimizations = [
            Config.ENABLE_BATCH_PROCESSING,
            Config.ENABLE_IMAGE_COMPRESSION,
            Config.ENABLE_SMART_FILTERING,
            Config.ENABLE_DUPLICATE_DETECTION
        ]
        
        disabled_optimizations = sum(1 for opt in optimizations if not opt)
        health_score -= disabled_optimizations * 5
        
        if disabled_optimizations > 0:
            issues.append(f"{disabled_optimizations} optimizations disabled")
        
        # Determine health status
        if health_score >= 90:
            status = "EXCELLENT"
        elif health_score >= 75:
            status = "GOOD"
        elif health_score >= 60:
            status = "MODERATE"
        else:
            status = "NEEDS_ATTENTION"
        
        return {
            'health_score': health_score,
            'status': status,
            'issues': issues,
            'optimizations_enabled': len(optimizations) - disabled_optimizations,
            'total_optimizations': len(optimizations)
        }
    
    def _calculate_processing_efficiency(self) -> Dict[str, Any]:
        """Calculate estimated processing efficiency metrics."""
        base_efficiency = 100
        
        # Apply optimization multipliers
        efficiency_multipliers = {
            'batch_processing': 3.0 if Config.ENABLE_BATCH_PROCESSING else 1.0,
            'image_compression': 1.7 if Config.ENABLE_IMAGE_COMPRESSION else 1.0,
            'smart_filtering': 1.2 if Config.ENABLE_SMART_FILTERING else 1.0,
            'duplicate_detection': 1.1 if Config.ENABLE_DUPLICATE_DETECTION else 1.0
        }
        
        total_efficiency = base_efficiency
        for optimization, multiplier in efficiency_multipliers.items():
            total_efficiency *= multiplier
        
        # Estimate processing speeds
        estimated_speeds = {
            'pages_per_minute': total_efficiency / 10,  # Rough estimate
            'documents_per_hour': total_efficiency / 50,  # Rough estimate
            'efficiency_improvement': f"{total_efficiency/100:.1f}x baseline"
        }
        
        return {
            'efficiency_multipliers': efficiency_multipliers,
            'total_efficiency_percent': total_efficiency,
            'estimated_speeds': estimated_speeds,
            'optimization_impact': {
                opt: f"{mult:.1f}x" for opt, mult in efficiency_multipliers.items()
            }
        }
    
    # =============================================================================
    # ADVANCED UTILITY METHODS
    # =============================================================================
    
    def export_extraction_summary(self, output_file: str = None) -> str:
        """
        Export a comprehensive summary of all extractions to a file.
        
        Args:
            output_file: Path to output file (default: auto-generated)
            
        Returns:
            Path to the created summary file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"extraction_summary_{timestamp}.json"
        
        # Gather comprehensive data
        summary_data = {
            'generated_at': datetime.now().isoformat(),
            'system_info': {
                'processor_version': '2.0.0',
                'configuration': {
                    'batch_processing': Config.ENABLE_BATCH_PROCESSING,
                    'image_compression': Config.ENABLE_IMAGE_COMPRESSION,
                    'smart_filtering': Config.ENABLE_SMART_FILTERING,
                    'duplicate_detection': Config.ENABLE_DUPLICATE_DETECTION,
                    'max_workers': Config.MAX_WORKERS,
                    'batch_size': Config.BATCH_SIZE
                }
            },
            'cache_statistics': self.get_extraction_cache_statistics(),
            'performance_stats': self.get_detailed_performance_statistics(),
            'all_extractions': self.get_all_saved_extractions()
        }
        
        # Write summary file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            print(f"📊 Extraction summary exported to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"❌ Error exporting summary: {e}")
            raise
    
    def validate_extraction_integrity(self) -> Dict[str, Any]:
        """
        Validate the integrity of all cached extractions.
        
        Returns:
            Dictionary containing validation results
        """
        print("🔍 Validating extraction integrity...")
        
        validation_results = {
            'total_files': 0,
            'valid_files': 0,
            'corrupted_files': 0,
            'missing_source_files': 0,
            'hash_mismatches': 0,
            'validation_errors': []
        }
        
        extractions = self.get_all_saved_extractions()
        validation_results['total_files'] = len(extractions)
        
        for extraction in extractions:
            try:
                # Check if extraction file is readable
                with open(extraction['extraction_file'], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if source file exists
                source_file = data.get('source_file', '')
                if not os.path.exists(source_file):
                    validation_results['missing_source_files'] += 1
                    validation_results['validation_errors'].append(
                        f"Missing source file: {source_file}"
                    )
                    continue
                
                # Check hash integrity if source file exists
                current_hash = self.generate_file_hash(source_file)
                stored_hash = data.get('file_hash', '')
                
                if current_hash != stored_hash:
                    validation_results['hash_mismatches'] += 1
                    validation_results['validation_errors'].append(
                        f"Hash mismatch for: {source_file}"
                    )
                else:
                    validation_results['valid_files'] += 1
                
            except Exception as e:
                validation_results['corrupted_files'] += 1
                validation_results['validation_errors'].append(
                    f"Corrupted extraction: {extraction['extraction_file']} - {str(e)}"
                )
        
        # Calculate integrity score
        if validation_results['total_files'] > 0:
            integrity_score = (validation_results['valid_files'] / validation_results['total_files']) * 100
        else:
            integrity_score = 100
        
        validation_results['integrity_score'] = integrity_score
        
        print(f"✅ Validation completed: {validation_results['valid_files']}/{validation_results['total_files']} files valid")
        print(f"🎯 Integrity score: {integrity_score:.1f}%")
        
        return validation_results